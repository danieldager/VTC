#!/usr/bin/env python3
"""Package pipeline — tile full audio files into WebDataset shards.

This is the CLI entry point.  The actual work is done by
:class:`~src.packaging.packer.Packer`.

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

import polars as pl

from src.packaging.packer import Packer
from src.packaging.stats import load_all_stats, save_all_stats
from src.plotting.figures import save_all_figures
from src.utils import (
    add_sample_argument,
    get_dataset_paths,
    load_manifest,
    sample_manifest,
)


# ---------------------------------------------------------------------------
# VAD vs VTC comparison (diagnostic — not part of packing itself)
# ---------------------------------------------------------------------------


def _run_comparison(ds, has_vad: bool) -> None:
    """Run VAD vs VTC comparison diagnostics (no-op if data missing)."""
    vtc_dir_check = ds.output / "vtc_raw"
    if not (has_vad and vtc_dir_check.exists()):
        return

    from src.plotting.compare import (
        calculate_metrics,
        diagnose_low_iou,
        load_parquet_dir,
        plot_dashboard,
        print_summary,
    )

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("dataset", help="Dataset name (must have VTC + VAD outputs).")
    parser.add_argument(
        "--max_clip", type=float, default=600.0,
        help="Maximum clip duration in seconds (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--split_search", type=float, default=120.0,
        help="Search window for finding split points (seconds, default: 120).",
    )
    parser.add_argument(
        "--audio_fmt", choices=["flac", "wav"], default="wav",
        help="Audio format in shards (default: wav).",
    )
    parser.add_argument(
        "--shard_size", type=int, default=100,
        help="Max clips per shard (default: 100).",
    )
    parser.add_argument(
        "--target_sr", type=int, default=16_000,
        help="Target sample rate (default: 16000).",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Skip processing; regenerate dashboard figures from cached stats.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip stats computation and figure rendering after packaging.",
    )
    add_sample_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ds = get_dataset_paths(args.dataset)
    print(f"Package pipeline: {args.dataset}")
    print(f"  output      : {ds.output}")

    fig_dir = Path("figures") / args.dataset

    # ------------------------------------------------------------------ #
    # Fast path: regenerate figures from cached stats only                #
    # ------------------------------------------------------------------ #
    if args.figures_only:
        print("  mode        : figures-only (loading cached stats)")
        print(f"  fig_dir     : {fig_dir}")
        dfs, tier_counts = load_all_stats(ds.output)
        print("Rendering figures ...", flush=True)
        save_all_figures(
            dfs, tier_counts, fig_dir,
            esc_stats_dir=ds.output / "esc_stats",
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
    print(f"  files       : {n_files}")

    if n_files == 0:
        print("Nothing to process.")
        sys.exit(0)

    # --- Check VTC exists (required) ---
    vtc_dir = ds.output / "vtc_merged"
    if not vtc_dir.exists():
        print(f"ERROR: {vtc_dir} not found. Run VTC first.", file=sys.stderr)
        sys.exit(1)

    # --- Build, write shards, write manifest ---
    packer = Packer(
        output_dir=ds.output,
        manifest_df=manifest_df,
        max_clip_s=args.max_clip,
        split_search_s=args.split_search,
        audio_fmt=args.audio_fmt,
        shard_size=args.shard_size,
        target_sr=args.target_sr,
    )

    # VAD vs VTC comparison (diagnostic)
    _run_comparison(ds, packer.has_vad)

    packer.build_clips()

    if not packer.all_clips:
        print("No clips to write.")
        sys.exit(0)

    shard_dir = ds.output / "shards"
    packer.write_shards(shard_dir)
    packer.write_manifest(shard_dir)

    # --- Stats & figures (optional) ---
    if not args.no_figures:
        t_stats = time.time()
        dfs = save_all_stats(packer.all_clips, ds.output, packer.tier_counts)
        print(f"  Stats computed in {time.time() - t_stats:.1f}s")

        print("Rendering figures...", flush=True)
        t_dash = time.time()
        save_all_figures(
            dfs, packer.tier_counts, fig_dir,
            esc_stats_dir=ds.output / "esc_stats",
        )
        print(f"  Dashboard rendered in {time.time() - t_dash:.1f}s")

    print("Done.")


if __name__ == "__main__":
    main()
