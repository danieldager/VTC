#!/usr/bin/env python3
"""Package pipeline — build WebDataset shards from VTC + VAD outputs.

Clips are built from **VTC segments** (the primary speech signal).
VAD segments are assigned to each clip as supplementary metadata for
comparison / agreement analysis.

Usage:
    python -m src.pipeline.package seedlings --sample 0.01
    python -m src.pipeline.package seedlings --max_gap 10 --max_clip 600
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.packaging.clips import Clip, Segment, build_clips
from src.packaging.writer import write_shards
from src.utils import (
    add_sample_argument,
    get_dataset_paths,
    load_manifest,
    sample_manifest,
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
            segments.append(Segment(
                onset=row["onset"],
                offset=row["offset"],
                label=row.get("label"),
            ))
    return sorted(segments, key=lambda s: s.onset)


def _get_file_duration(metadata_dir: Path, uid: str) -> float | None:
    """Get audio duration from VAD metadata."""
    meta_path = metadata_dir / "metadata.parquet"
    if not meta_path.exists():
        return None
    df = pl.read_parquet(meta_path).filter(pl.col("file_id") == uid)
    if df.is_empty():
        return None
    return float(df["duration"][0])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _print_stats(clips: list[tuple[str, Path, int, Clip]]) -> dict:
    """Print comprehensive clip statistics.  Returns stats dict for figure."""
    all_clips = [c for _, _, _, c in clips]
    n = len(all_clips)

    durations = np.array([c.duration for c in all_clips])
    vtc_durs = np.array([c.vtc_speech_duration for c in all_clips])
    vad_durs = np.array([c.vad_speech_duration for c in all_clips])
    vtc_dens = np.array([c.speech_density for c in all_clips])
    vad_dens = np.array([c.vad_density for c in all_clips])
    ious = np.array([c.vad_vtc_iou for c in all_clips])
    turns = np.array([c.n_turns for c in all_clips])
    n_labels = np.array([c.n_labels for c in all_clips])
    has_adult = np.array([c.has_adult for c in all_clips])

    # Per-segment stats (across all clips)
    all_vtc_seg_durs = [s.duration for c in all_clips for s in c.vtc_segments]
    all_vtc_gaps = []
    for c in all_clips:
        segs = sorted(c.vtc_segments, key=lambda s: s.onset)
        for i in range(1, len(segs)):
            gap = segs[i].onset - segs[i - 1].offset
            if gap > 0:
                all_vtc_gaps.append(gap)

    vtc_seg_durs = np.array(all_vtc_seg_durs) if all_vtc_seg_durs else np.array([0.0])
    vtc_gaps = np.array(all_vtc_gaps) if all_vtc_gaps else np.array([0.0])

    def _fmt(arr: np.ndarray) -> str:
        return (f"mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
                f"std={arr.std():.1f}  min={arr.min():.1f}  max={arr.max():.1f}")

    print(f"\n{'═' * 60}")
    print(f"  CLIP STATISTICS  ({n} clips)")
    print(f"{'═' * 60}")

    print(f"\n  Clip duration (s):")
    print(f"    {_fmt(durations)}")
    print(f"    total = {durations.sum() / 3600:.1f}h")

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
    print(f"    {_fmt(n_labels.astype(float))}")
    print(f"    clips with adult (FEM/MAL): "
          f"{has_adult.sum()}/{n} ({has_adult.mean():.0%})")

    print(f"\n  VTC segment duration (s):")
    print(f"    {_fmt(vtc_seg_durs)}")
    print(f"    total segments: {len(all_vtc_seg_durs)}")

    print(f"\n  Gap between VTC segments (s):")
    print(f"    {_fmt(vtc_gaps)}")
    print(f"    total gaps: {len(all_vtc_gaps)}")

    print(f"{'═' * 60}\n")

    return {
        "durations": durations,
        "vtc_durs": vtc_durs,
        "vad_durs": vad_durs,
        "vtc_dens": vtc_dens,
        "vad_dens": vad_dens,
        "ious": ious,
        "turns": turns,
        "n_labels": n_labels,
        "has_adult": has_adult,
        "vtc_seg_durs": vtc_seg_durs,
        "vtc_gaps": vtc_gaps,
    }


def _save_figure(stats: dict, output_path: Path) -> None:
    """Save a multi-panel summary figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Clip Packaging Summary", fontsize=14, fontweight="bold")

    # 1. Clip duration distribution
    ax = axes[0, 0]
    ax.hist(stats["durations"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Clip Duration Distribution")
    ax.axvline(np.median(stats["durations"]), color="red", ls="--", lw=1,
               label=f'median={np.median(stats["durations"]):.0f}s')
    ax.legend(fontsize=8)

    # 2. VTC speech density
    ax = axes[0, 1]
    ax.hist(stats["vtc_dens"], bins=40, color="#55A868", edgecolor="white", alpha=0.8)
    ax.set_xlabel("VTC speech density")
    ax.set_ylabel("Count")
    ax.set_title("VTC Speech Density")
    ax.axvline(np.median(stats["vtc_dens"]), color="red", ls="--", lw=1,
               label=f'median={np.median(stats["vtc_dens"]):.2f}')
    ax.legend(fontsize=8)

    # 3. VAD vs VTC density scatter
    ax = axes[0, 2]
    ax.scatter(stats["vtc_dens"], stats["vad_dens"], alpha=0.3, s=10, c="#C44E52")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("VTC density")
    ax.set_ylabel("VAD density")
    ax.set_title("VAD vs VTC Density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 4. IoU distribution
    ax = axes[1, 0]
    ax.hist(stats["ious"], bins=40, color="#8172B2", edgecolor="white", alpha=0.8)
    ax.set_xlabel("VAD–VTC IoU")
    ax.set_ylabel("Count")
    ax.set_title("VAD–VTC Agreement (IoU)")
    ax.axvline(np.median(stats["ious"]), color="red", ls="--", lw=1,
               label=f'median={np.median(stats["ious"]):.2f}')
    ax.legend(fontsize=8)

    # 5. Turns per clip
    ax = axes[1, 1]
    max_turns = int(min(stats["turns"].max(), 50))
    ax.hist(stats["turns"], bins=range(0, max_turns + 2), color="#CCB974",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Speaker turns")
    ax.set_ylabel("Count")
    ax.set_title("Speaker Turns per Clip")
    ax.axvline(np.median(stats["turns"]), color="red", ls="--", lw=1,
               label=f'median={np.median(stats["turns"]):.0f}')
    ax.legend(fontsize=8)

    # 6. Labels per clip
    ax = axes[1, 2]
    ax.hist(stats["n_labels"], bins=range(0, 6), color="#64B5CD",
            edgecolor="white", alpha=0.8, align="left")
    ax.set_xlabel("Unique labels")
    ax.set_ylabel("Count")
    ax.set_title("Label Diversity per Clip")
    ax.set_xticks(range(0, 5))

    # 7. VTC segment duration
    ax = axes[2, 0]
    clipped = np.clip(stats["vtc_seg_durs"], 0, 30)
    ax.hist(clipped, bins=60, color="#DD8452", edgecolor="white", alpha=0.8)
    ax.set_xlabel("VTC segment duration (s, capped at 30)")
    ax.set_ylabel("Count")
    ax.set_title("VTC Segment Duration")
    ax.axvline(np.median(stats["vtc_seg_durs"]), color="red", ls="--", lw=1,
               label=f'median={np.median(stats["vtc_seg_durs"]):.1f}s')
    ax.legend(fontsize=8)

    # 8. Gap between VTC segments
    ax = axes[2, 1]
    clipped_gaps = np.clip(stats["vtc_gaps"], 0, 15)
    ax.hist(clipped_gaps, bins=60, color="#DA8BC3", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Gap between VTC segments (s, capped at 15)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-segment Gap")
    ax.axvline(np.median(stats["vtc_gaps"]), color="red", ls="--", lw=1,
               label=f'median={np.median(stats["vtc_gaps"]):.1f}s')
    ax.legend(fontsize=8)

    # 9. Speech duration vs clip duration
    ax = axes[2, 2]
    ax.scatter(stats["durations"], stats["vtc_durs"], alpha=0.3, s=10, c="#4C72B0")
    ax.plot([0, stats["durations"].max()], [0, stats["durations"].max()],
            "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("VTC speech (s)")
    ax.set_title("Speech vs Clip Duration")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
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
        "--max_clip", type=float, default=600.0,
        help="Maximum clip duration in seconds (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--buffer", type=float, default=5.0,
        help="Non-speech padding on each side (seconds, default: 5).",
    )
    parser.add_argument(
        "--max_gap", type=float, default=10.0,
        help="Max VTC silence gap before splitting regions (seconds, default: 10).",
    )
    parser.add_argument(
        "--split_search", type=float, default=120.0,
        help="Search window for finding split points (seconds, default: 120).",
    )
    parser.add_argument(
        "--min_speech", type=float, default=5.0,
        help="Discard clips with less VTC speech than this (seconds, default: 5).",
    )
    parser.add_argument(
        "--min_seg", type=float, default=0.5,
        help="Filter VTC segments shorter than this (seconds, default: 0.5).",
    )
    parser.add_argument(
        "--audio_fmt", choices=["flac", "wav"], default="flac",
        help="Audio format in shards (default: flac).",
    )
    parser.add_argument(
        "--shard_size", type=int, default=100,
        help="Max clips per shard (default: 100).",
    )
    parser.add_argument(
        "--target_sr", type=int, default=16_000,
        help="Target sample rate (default: 16000).",
    )
    add_sample_argument(parser)
    args = parser.parse_args()

    ds = get_dataset_paths(args.dataset)
    print(f"Package pipeline: {args.dataset}")
    print(f"  output   : {ds.output}")
    print(f"  audio_fmt: {args.audio_fmt}")
    print(f"  max_clip : {args.max_clip:.0f}s")
    print(f"  buffer   : {args.buffer:.0f}s")
    print(f"  max_gap  : {args.max_gap:.0f}s")
    print(f"  min_seg  : {args.min_seg:.1f}s")

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

    # --- Build clips for each file ---
    t0 = time.time()
    all_clips: list[tuple[str, Path, int, Clip]] = []
    skipped = 0

    for row in manifest_df.iter_rows(named=True):
        audio_path = Path(row["path"])
        uid = audio_path.stem

        vtc_segs = _load_vtc_segments(ds.output, uid)
        vad_segs = _load_vad_segments(ds.output, uid) if has_vad else []
        file_dur = _get_file_duration(ds.metadata, uid)

        if file_dur is None:
            print(f"  WARN: no metadata for {uid}, skipping")
            skipped += 1
            continue

        if not vtc_segs:
            print(f"  WARN: no VTC segments for {uid}, skipping")
            skipped += 1
            continue

        clips = build_clips(
            vtc_segments=vtc_segs,
            vad_segments=vad_segs,
            file_duration=file_dur,
            max_clip_s=args.max_clip,
            buffer_s=args.buffer,
            max_gap=args.max_gap,
            split_search_s=args.split_search,
            min_clip_speech_s=args.min_speech,
            min_seg_s=args.min_seg,
        )

        for idx, clip in enumerate(clips):
            all_clips.append((uid, audio_path, idx, clip))

    clip_time = time.time() - t0
    print(f"\nClip building: {len(all_clips)} clips from {n_files - skipped} files "
          f"in {clip_time:.1f}s")
    if skipped:
        print(f"  skipped: {skipped} files")

    if not all_clips:
        print("No clips to write.")
        sys.exit(0)

    # --- Print stats & save figure ---
    stats = _print_stats(all_clips)
    fig_dir = Path("figures") / args.dataset
    _save_figure(stats, fig_dir / "clip_summary.png")

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
        manifest_rows.append(meta)

    clip_manifest = pl.DataFrame(manifest_rows)
    manifest_path = shard_dir / "manifest.csv"
    clip_manifest.write_csv(manifest_path)
    print(f"  Manifest: {manifest_path}  ({len(clip_manifest)} clips)")

    total_time = clip_time + write_time
    print(f"\nDone in {total_time:.0f}s")


if __name__ == "__main__":
    main()
