#!/usr/bin/env python3
"""Package pipeline — tile full audio files into WebDataset shards.

All audio is preserved.  VAD + VTC segments are used only to find safe
cut points (never cutting during speech/vocalisation).

Usage:
    python -m src.pipeline.package seedlings --sample 0.01
    python -m src.pipeline.package seedlings --max_clip 600
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.core import LABEL_COLORS, VTC_LABELS
from src.packaging.clips import CUT_TIERS, Clip, Segment, build_clips
from src.packaging.stats import save_all_stats, load_all_stats
from src.packaging.writer import write_shards
from src.plotting.figures import save_all_figures
from src.plotting.packaging import save_figure, save_label_figures
from src.utils import (
    add_sample_argument,
    get_dataset_paths,
    load_manifest,
    sample_manifest,
)
from src.pipeline.compare import (
    calculate_metrics,
    diagnose_low_iou,
    load_parquet_dir,
    plot_dashboard,
    print_summary,
)


# ---------------------------------------------------------------------------
# Segment loading
# ---------------------------------------------------------------------------


def _load_vad_segments(output_dir: Path, uid: str) -> list[Segment]:
    """Load merged VAD segments for one file."""
    seg_path = output_dir / "vad_merged" / "segments.parquet"
    if not seg_path.exists():
        return []
    df = pl.read_parquet(seg_path).filter(pl.col("uid") == uid)
    return [
        Segment(onset=row["onset"], offset=row["offset"])
        for row in df.iter_rows(named=True)
    ]


def _load_vtc_segments(output_dir: Path, uid: str) -> list[Segment]:
    """Load merged VTC segments (with labels) for one file."""
    vtc_dir = output_dir / "vtc_merged"
    if not vtc_dir.exists():
        return []
    segments: list[Segment] = []
    for p in sorted(vtc_dir.glob("*.parquet")):
        df = pl.read_parquet(p).filter(pl.col("uid") == uid)
        for row in df.iter_rows(named=True):
            segments.append(
                Segment(
                    onset=row["onset"],
                    offset=row["offset"],
                    label=row.get("label"),
                )
            )
    return sorted(segments, key=lambda s: s.onset)


def _get_file_duration(output_dir: Path, uid: str) -> float | None:
    """Get audio duration from VAD metadata."""
    meta_path = output_dir / "vad_meta" / "metadata.parquet"
    if not meta_path.exists():
        return None
    df = pl.read_parquet(meta_path).filter(pl.col("file_id") == uid)
    if df.is_empty():
        return None
    return float(df["duration"][0])


def _load_noise(
    output_dir: Path,
    uid: str,
) -> tuple[np.ndarray | None, list[str], float]:
    """Load pooled noise category probabilities for one file.

    Returns ``(categories_array, category_names, pool_step_s)`` or
    ``(None, [], 1.0)`` if the noise file is not found.
    """
    noise_path = output_dir / "noise" / f"{uid}.npz"
    if not noise_path.exists():
        return None, [], 1.0
    data = np.load(noise_path, allow_pickle=True)
    cats = data["categories"].astype(np.float32)  # (n_bins, n_cats)
    cat_names = list(data["category_names"])  # list[str]
    pool_step_s = float(data["pool_step_s"])
    return cats, cat_names, pool_step_s


def _slice_noise_for_clip(
    file_noise: np.ndarray,
    pool_step_s: float,
    abs_onset: float,
    abs_offset: float,
) -> np.ndarray:
    """Slice file-level pooled noise categories for one clip.

    Returns a float16 array of shape (n_bins, n_cats).
    """
    start_idx = int(abs_onset / pool_step_s)
    end_idx = int(np.ceil(abs_offset / pool_step_s))
    start_idx = max(0, min(start_idx, file_noise.shape[0]))
    end_idx = max(start_idx, min(end_idx, file_noise.shape[0]))
    return file_noise[start_idx:end_idx].astype(np.float16)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _print_stats(
    clips: list[tuple[str, Path, int, Clip]],
    tier_counts: dict[str, int] | None = None,
) -> dict:
    """Print comprehensive clip statistics.  Returns stats dict for figures."""
    all_clips = [c for _, _, _, c in clips]
    n = len(all_clips)

    durations = np.array([c.duration for c in all_clips])
    vtc_durs = np.array([c.vtc_speech_duration for c in all_clips])
    vad_durs = np.array([c.vad_speech_duration for c in all_clips])
    vtc_dens = np.array([c.speech_density for c in all_clips])
    vad_dens = np.array([c.vad_density for c in all_clips])
    ious = np.array([c.vad_vtc_iou for c in all_clips])
    turns = np.array([c.n_turns for c in all_clips])
    n_labels_arr = np.array([c.n_labels for c in all_clips])
    has_adult = np.array([c.has_adult for c in all_clips])
    child_fracs = np.array([c.child_fraction for c in all_clips])
    dominant_labels = [c.dominant_label or "?" for c in all_clips]

    # Per-label segment data
    label_seg_durs: dict[str, list[float]] = {l: [] for l in VTC_LABELS}
    label_seg_counts: dict[str, int] = {l: 0 for l in VTC_LABELS}
    for c in all_clips:
        for s in c.vtc_segments:
            if s.label in VTC_LABELS:
                label_seg_durs[s.label].append(s.duration)
                label_seg_counts[s.label] += 1

    # Per-label VAD coverage (per-clip)
    label_vad_coverage: dict[str, list[float]] = {l: [] for l in VTC_LABELS}
    for c in all_clips:
        cov = c.vad_coverage_by_label()
        for l in VTC_LABELS:
            if l in cov:
                label_vad_coverage[l].append(cov[l])

    # Gap analysis: same-label vs cross-label
    same_label_gaps: list[float] = []
    cross_label_gaps: list[float] = []
    gap_label_pairs: list[tuple[str, str, float]] = []
    for c in all_clips:
        segs = sorted(c.vtc_segments, key=lambda s: s.onset)
        for i in range(1, len(segs)):
            gap = segs[i].onset - segs[i - 1].offset
            if gap > 0:
                l_from = segs[i - 1].label or "?"
                l_to = segs[i].label or "?"
                gap_label_pairs.append((l_from, l_to, gap))
                if l_from == l_to:
                    same_label_gaps.append(gap)
                else:
                    cross_label_gaps.append(gap)

    all_vtc_seg_durs = [s.duration for c in all_clips for s in c.vtc_segments]
    all_vtc_gaps = same_label_gaps + cross_label_gaps

    vtc_seg_durs = np.array(all_vtc_seg_durs) if all_vtc_seg_durs else np.array([0.0])
    vtc_gaps = np.array(all_vtc_gaps) if all_vtc_gaps else np.array([0.0])

    def _fmt(arr: np.ndarray) -> str:
        return (
            f"mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
            f"std={arr.std():.1f}  min={arr.min():.1f}  max={arr.max():.1f}"
        )

    print(f"\n{'═' * 60}")
    print(f"  CLIP STATISTICS  ({n} clips)")
    print(f"{'═' * 60}")

    print(f"\n  Clip duration (s):")
    print(f"    {_fmt(durations)}")
    print(f"    total = {durations.sum() / 3600:.1f}h")

    # --- Cut-point tier breakdown ---
    if tier_counts is not None:
        total_cuts = sum(tier_counts.values())
        if total_cuts > 0:
            _tier_labels = {
                "long_union_gap": "1. Long silence gap (VAD∪VTC)",
                "short_union_gap": "2. Short silence gap (VAD∪VTC)",
                "vad_only_gap": "3. VAD-only gap (VTC active)",
                "vtc_only_gap": "4. VTC-only gap (VAD active)",
                "speaker_boundary": "5. Speaker boundary (active audio)",
                "hard_cut": "6. Hard cut (no gaps/boundaries)",
                "degenerate_window": "⚠  Degenerate window (forced)",
            }
            print(f"\n  Cut-point tiers ({total_cuts} cuts):")
            for tier_key, label in _tier_labels.items():
                cnt = tier_counts.get(tier_key, 0)
                if cnt > 0:
                    pct = 100.0 * cnt / total_cuts
                    print(f"    {label:44s} {cnt:>4d}  ({pct:5.1f}%)")
            silent = tier_counts.get("long_union_gap", 0) + tier_counts.get(
                "short_union_gap", 0
            )
            print(f"    {'─' * 55}")
            print(
                f"    {'Clean (silence gap):':44s} {silent:>4d}  "
                f"({100.0 * silent / total_cuts:5.1f}%)"
            )
            degraded = (
                tier_counts.get("speaker_boundary", 0)
                + tier_counts.get("hard_cut", 0)
                + tier_counts.get("degenerate_window", 0)
            )
            if degraded:
                print(
                    f"    {'Degraded (boundary/hard/degen):':44s} "
                    f"{degraded:>4d}  "
                    f"({100.0 * degraded / total_cuts:5.1f}%)"
                )

    print(f"\n  VTC speech per clip (s):")
    print(f"    {_fmt(vtc_durs)}")
    print(f"    total = {vtc_durs.sum() / 3600:.1f}h")

    print(f"\n  VTC speech density:")
    print(f"    {_fmt(vtc_dens)}")

    print(f"\n  VAD speech per clip (s):")
    print(f"    {_fmt(vad_durs)}")
    print(f"    total = {vad_durs.sum() / 3600:.1f}h")

    print(f"\n  VAD speech density:")
    print(f"    {_fmt(vad_dens)}")

    print(f"\n  VAD–VTC agreement (IoU per clip):")
    print(f"    {_fmt(ious)}")

    print(f"\n  Speaker turns per clip:")
    print(f"    {_fmt(turns.astype(float))}")

    print(f"\n  Labels per clip:")
    print(f"    {_fmt(n_labels_arr.astype(float))}")
    print(
        f"    clips with adult (FEM/MAL): "
        f"{has_adult.sum()}/{n} ({has_adult.mean():.0%})"
    )

    print(f"\n  VTC segment duration (s):")
    print(f"    {_fmt(vtc_seg_durs)}")
    print(f"    total segments: {len(all_vtc_seg_durs)}")

    print(f"\n  Gap between VTC segments (s):")
    print(f"    {_fmt(vtc_gaps)}")
    print(f"    total gaps: {len(all_vtc_gaps)}")

    # --- Per-label stats ---
    print(f"\n{'─' * 60}")
    print(f"  PER-LABEL STATISTICS")
    print(f"{'─' * 60}")
    for l in VTC_LABELS:
        durs = label_seg_durs[l]
        if not durs:
            print(f"\n  {l}: no segments")
            continue
        arr = np.array(durs)
        total_h = arr.sum() / 3600
        cov = label_vad_coverage[l]
        cov_str = (
            f"  VAD coverage: mean={np.mean(cov):.2f}  std={np.std(cov):.2f}"
            if cov
            else "  VAD coverage: N/A"
        )
        print(f"\n  {l}: {label_seg_counts[l]} segments, {total_h:.2f}h total")
        print(f"    segment duration: {_fmt(arr)}")
        print(f"    {cov_str}")

    # --- Gap analysis ---
    print(f"\n{'─' * 60}")
    print(f"  GAP ANALYSIS (same-label vs cross-label)")
    print(f"{'─' * 60}")
    sl_arr = np.array(same_label_gaps) if same_label_gaps else np.array([0.0])
    cl_arr = np.array(cross_label_gaps) if cross_label_gaps else np.array([0.0])
    print(f"\n  Same-label gaps ({len(same_label_gaps)}):")
    print(f"    {_fmt(sl_arr)}")
    print(f"\n  Cross-label gaps ({len(cross_label_gaps)}):")
    print(f"    {_fmt(cl_arr)}")

    # --- Child vs Adult ---
    print(f"\n{'─' * 60}")
    print(f"  CHILD vs ADULT")
    print(f"{'─' * 60}")
    print(f"\n  Child fraction per clip: {_fmt(child_fracs)}")
    child_dom = sum(1 for d in dominant_labels if d in ("KCHI", "OCH"))
    adult_dom = sum(1 for d in dominant_labels if d in ("FEM", "MAL"))
    print(f"  Dominant label: child={child_dom}  adult={adult_dom}")

    print(f"{'═' * 60}\n")

    return {
        "durations": durations,
        "vtc_durs": vtc_durs,
        "vad_durs": vad_durs,
        "vtc_dens": vtc_dens,
        "vad_dens": vad_dens,
        "ious": ious,
        "turns": turns,
        "n_labels": n_labels_arr,
        "has_adult": has_adult,
        "vtc_seg_durs": vtc_seg_durs,
        "vtc_gaps": vtc_gaps,
        "child_fracs": child_fracs,
        "dominant_labels": dominant_labels,
        "label_seg_durs": {
            l: np.array(d) if d else np.array([]) for l, d in label_seg_durs.items()
        },
        "label_seg_counts": label_seg_counts,
        "label_vad_coverage": {
            l: np.array(d) if d else np.array([]) for l, d in label_vad_coverage.items()
        },
        "same_label_gaps": (
            np.array(same_label_gaps) if same_label_gaps else np.array([])
        ),
        "cross_label_gaps": (
            np.array(cross_label_gaps) if cross_label_gaps else np.array([])
        ),
        "gap_label_pairs": gap_label_pairs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        format="%(levelname)s [%(name)s] %(message)s",
        level=logging.WARNING,
    )
    parser = argparse.ArgumentParser(
        description="Package audio clips into WebDataset tar shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.pipeline.package seedlings --sample 0.01\n"
            "  python -m src.pipeline.package seedlings --audio_fmt wav\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (must have VTC + VAD outputs).",
    )
    parser.add_argument(
        "--max_clip",
        type=float,
        default=600.0,
        help="Maximum clip duration in seconds (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--split_search",
        type=float,
        default=120.0,
        help="Search window for finding split points (seconds, default: 120).",
    )
    parser.add_argument(
        "--audio_fmt",
        choices=["flac", "wav"],
        default="wav",
        help="Audio format in shards (default: wav).",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100,
        help="Max clips per shard (default: 100).",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16_000,
        help="Target sample rate (default: 16000).",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Skip all processing. Load cached stats from output/stats/ and "
            "regenerate dashboard figures only."
        ),
    )
    add_sample_argument(parser)
    args = parser.parse_args()

    ds = get_dataset_paths(args.dataset)
    print(f"Package pipeline: {args.dataset}")
    print(f"  output      : {ds.output}")

    # ------------------------------------------------------------------ #
    # Fast path: regenerate figures from cached stats only                #
    # ------------------------------------------------------------------ #
    if args.figures_only:
        print("  mode        : figures-only (loading cached stats)")
        fig_dir = Path("figures") / args.dataset
        print(f"  fig_dir     : {fig_dir}")
        dfs, tier_counts = load_all_stats(ds.output)
        print("Rendering figures ...", flush=True)
        save_all_figures(
            dfs, tier_counts, fig_dir,
            noise_stats_dir=ds.output / "noise_stats",
        )
        print("Done.")
        return

    print(f"  audio_fmt   : {args.audio_fmt}")
    print(f"  max_clip    : {args.max_clip:.0f}s")
    print(f"  split_search: {args.split_search:.0f}s")

    # --- Load manifest ---
    manifest_df = load_manifest(ds.manifest)
    manifest_df = sample_manifest(manifest_df, args.sample)
    n_files = len(manifest_df)
    print(f"  files    : {n_files}")

    if n_files == 0:
        print("Nothing to process.")
        sys.exit(0)

    # --- Check outputs exist ---
    vtc_dir = ds.output / "vtc_merged"
    if not vtc_dir.exists():
        print(f"ERROR: {vtc_dir} not found. Run VTC first.", file=sys.stderr)
        sys.exit(1)

    # VAD is optional (for comparison metadata)
    vad_merged = ds.output / "vad_merged" / "segments.parquet"
    has_vad = vad_merged.exists()
    if not has_vad:
        print("  WARN: no VAD segments — clips will lack VAD comparison data")

    # SNR data is now handled via segment_snr parquet (not arrays)

    # Noise is optional (from PANNs pipeline)
    noise_dir = ds.output / "noise"
    has_noise = noise_dir.exists() and any(noise_dir.glob("*.npz"))
    if has_noise:
        print(f"  Noise data: {noise_dir}")
    else:
        print("  WARN: no noise data — clips will lack noise classification")

    # --- Compare VAD vs VTC (integrated from compare pipeline) ---
    vtc_dir_check = ds.output / "vtc_raw"
    if has_vad and vtc_dir_check.exists():
        print("\nVAD vs VTC comparison...")
        vad_raw_df = load_parquet_dir(ds.output / "vad_raw")
        vad_merged_df = load_parquet_dir(ds.output / "vad_merged")
        vtc_raw_df = load_parquet_dir(ds.output / "vtc_raw")
        vtc_merged_df = load_parquet_dir(ds.output / "vtc_merged")

        # Build per-file metadata from VAD metadata
        vad_meta_for_cmp = None
        meta_path_cmp = ds.output / "vad_meta" / "metadata.parquet"
        if meta_path_cmp.exists():
            vad_meta_full_cmp = pl.read_parquet(meta_path_cmp)
            vad_meta_for_cmp = vad_meta_full_cmp.select(
                pl.col("file_id").str.replace(r"\.wav$", "").alias("uid"),
                pl.col("duration").alias("file_total"),
            )

        vtc_meta_dir = ds.output / "vtc_meta"
        vtc_meta_thresh = None
        if vtc_meta_dir.exists():
            vtc_meta_files = sorted(vtc_meta_dir.glob("*.parquet"))
            if vtc_meta_files:
                vtc_meta_thresh = pl.read_parquet(vtc_meta_files).select(
                    "uid", "vtc_threshold"
                )

        low_thresh = 0.3
        results_raw = calculate_metrics(vad_raw_df, vtc_raw_df, vad_meta_for_cmp)
        if vtc_meta_thresh is not None:
            results_raw = results_raw.join(vtc_meta_thresh, on="uid", how="left")
        stats_raw = print_summary(results_raw, "RAW: VAD vs VTC", low_thresh)
        results_raw.write_csv(ds.output / "compare_raw.csv")

        results_merged = calculate_metrics(
            vad_merged_df, vtc_merged_df, vad_meta_for_cmp
        )
        if vtc_meta_thresh is not None:
            results_merged = results_merged.join(vtc_meta_thresh, on="uid", how="left")
        stats_merged = print_summary(results_merged, "MERGED: VAD vs VTC", low_thresh)
        results_merged.write_csv(ds.output / "compare_merged.csv")

        fig_dir_cmp = ds.figures / "vtc"
        fig_dir_cmp.mkdir(parents=True, exist_ok=True)
        if vad_meta_for_cmp is not None:
            results_raw = results_raw.join(
                vad_meta_for_cmp.select("uid", "file_total"), on="uid", how="left"
            )
            results_merged = results_merged.join(
                vad_meta_for_cmp.select("uid", "file_total"), on="uid", how="left"
            )

        plot_dashboard(
            results_raw,
            stats_raw,
            "RAW: VAD vs VTC",
            fig_dir_cmp / "compare_raw.png",
            low_thresh,
            target_iou=0.9,
        )
        plot_dashboard(
            results_merged,
            stats_merged,
            "MERGED: VAD vs VTC",
            fig_dir_cmp / "compare_merged.png",
            low_thresh,
            target_iou=0.9,
        )

        diag = diagnose_low_iou(results_merged, vtc_merged_df, low_thresh)
        diag.write_csv(ds.output / "diagnostics.csv")

    # --- Build clips for each file ---
    t0 = time.time()
    all_clips: list[tuple[str, Path, int, Clip]] = []
    global_tier_counts: dict[str, int] = {t: 0 for t in CUT_TIERS}
    skipped = 0

    for row in manifest_df.iter_rows(named=True):
        audio_path = Path(row["path"])
        uid = audio_path.stem

        vtc_segs = _load_vtc_segments(ds.output, uid)
        vad_segs = _load_vad_segments(ds.output, uid) if has_vad else []
        file_noise, noise_cats, noise_step = (
            _load_noise(ds.output, uid) if has_noise else (None, [], 1.0)
        )
        file_dur = _get_file_duration(ds.output, uid)

        if file_dur is None:
            print(f"  WARN: no metadata for {uid}, skipping")
            skipped += 1
            continue

        clips, tier_counts = build_clips(
            vtc_segments=vtc_segs,
            vad_segments=vad_segs,
            file_duration=file_dur,
            max_clip_s=args.max_clip,
            split_search_s=args.split_search,
        )
        for t, n in tier_counts.items():
            global_tier_counts[t] += n

        for idx, clip in enumerate(clips):
            if file_noise is not None:
                clip.noise_array = _slice_noise_for_clip(
                    file_noise,
                    noise_step,
                    clip.abs_onset,
                    clip.abs_offset,
                )
                clip.noise_categories = noise_cats
                clip.noise_step_s = noise_step
            all_clips.append((uid, audio_path, idx, clip))

    clip_time = time.time() - t0
    print(
        f"\nClip building: {len(all_clips)} clips from {n_files - skipped} files "
        f"in {clip_time:.1f}s"
    )
    if skipped:
        print(f"  skipped: {skipped} files")

    if not all_clips:
        print("No clips to write.")
        sys.exit(0)

    # --- Print stats & save figures ---
    stats = _print_stats(all_clips, global_tier_counts)
    fig_dir = Path("figures") / args.dataset
    (fig_dir / "overview").mkdir(parents=True, exist_ok=True)
    (fig_dir / "speech").mkdir(parents=True, exist_ok=True)
    save_figure(stats, fig_dir / "overview" / "clip_summary.png")
    save_label_figures(stats, fig_dir / "speech" / "label_analysis.png")

    # --- Build intermediate DataFrames & figures ---
    t_stats = time.time()
    dfs = save_all_stats(all_clips, ds.output, global_tier_counts)
    print(f"  Stats computed in {time.time() - t_stats:.1f}s")

    print("Rendering figures...", flush=True)
    t_dash = time.time()
    save_all_figures(
        dfs, global_tier_counts, fig_dir,
        noise_stats_dir=ds.output / "noise_stats",
    )
    print(f"  Dashboard rendered in {time.time() - t_dash:.1f}s")

    # --- Write shards ---
    print("Writing shards...", flush=True)
    t0 = time.time()
    shard_dir = ds.output / "shards"
    shard_paths = write_shards(
        clips=all_clips,
        output_dir=shard_dir,
        prefix="shards",
        max_shard_clips=args.shard_size,
        audio_fmt=args.audio_fmt,
        target_sr=args.target_sr,
    )
    write_time = time.time() - t0

    print(f"  Wrote {len(shard_paths)} shards in {write_time:.1f}s")
    for p in shard_paths:
        size_mb = p.stat().st_size / 1e6
        print(f"    {p.name}  ({size_mb:.1f} MB)")

    # --- Write manifest ---
    manifest_rows = []
    for uid, audio_path, clip_idx, clip in all_clips:
        meta = clip.to_metadata(uid, clip_idx)
        # Drop segment lists from CSV manifest (too large); keep in JSON
        meta.pop("vad_segments", None)
        meta.pop("vtc_segments", None)
        meta.pop("snr", None)  # array too large for CSV; kept in JSON
        meta.pop("c50", None)  # array too large for CSV; kept in JSON
        meta.pop("noise_profile", None)  # dict too large for CSV; kept in JSON
        # Flatten nested types for CSV compatibility
        meta["labels_present"] = ";".join(meta.get("labels_present", []))
        ld = meta.pop("label_durations", {})
        for lbl in VTC_LABELS:
            meta[f"dur_{lbl}"] = round(ld.get(lbl, 0.0), 3)
        vc = meta.pop("vad_coverage_by_label", {})
        for lbl in VTC_LABELS:
            meta[f"vad_cov_{lbl}"] = round(vc.get(lbl, 0.0), 3)
        manifest_rows.append(meta)

    clip_manifest = pl.DataFrame(manifest_rows)
    manifest_path = shard_dir / "manifest.csv"
    clip_manifest.write_csv(manifest_path)
    print(f"  Manifest: {manifest_path}  ({len(clip_manifest)} clips)")

    total_time = clip_time + write_time
    print(f"\nDone in {total_time:.0f}s")


if __name__ == "__main__":
    main()
