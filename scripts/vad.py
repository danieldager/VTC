#!/usr/bin/env python3
"""
TenVAD pipeline: voice activity detection with multiprocessing.

All paths derived from the dataset name:
    manifests/{dataset}.parquet          input manifest
    metadata/{dataset}/metadata.parquet  per-file metadata
    output/{dataset}/vad_raw/            raw VAD segments
    output/{dataset}/vad_merged/         merged VAD segments

Usage:
    python scripts/vad.py chunks30
    python scripts/vad.py chunks30 --sample 500
"""

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore", message=".*In 2.9.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")

from scripts.core.checkpoint import CKPT_META, CKPT_SEGS, clear_checkpoint
from scripts.core.parallel import run_vad_parallel
from scripts.core.vad_processing import set_seeds
from scripts.utils import (
    add_sample_argument,
    atomic_write_parquet,
    get_dataset_paths,
    load_manifest,
    log_benchmark,
    merge_segments_df,
    sample_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TenVAD pipeline: voice activity detection on audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/vad.py chunks30\n"
            "  python scripts/vad.py chunks30 --sample 500\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name — used to derive output/, metadata/, and figures/ directories.",
    )
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: all CPUs)",
    )
    add_sample_argument(parser)
    args = parser.parse_args()

    set_seeds(42)

    ds = get_dataset_paths(args.dataset)
    print(f"Dataset:   {args.dataset}")
    print(f"  manifest : {ds.manifest}")
    print(f"  output   : {ds.output}")

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    manifest_df = load_manifest(ds.manifest)
    manifest_df = sample_manifest(manifest_df, args.sample)
    if args.sample is not None:
        print(f"  sample   : {len(manifest_df)} files")

    # ------------------------------------------------------------------
    # Resolve workers + NFS I/O safeguard
    # ------------------------------------------------------------------
    workers = args.workers or mp.cpu_count()

    wavs_preview = [
        Path(p) for p in manifest_df["path"].drop_nulls().head(20).to_list()
    ]
    if wavs_preview:
        sample_sizes = [w.stat().st_size for w in wavs_preview[:10] if w.exists()]
        if sample_sizes:
            median_size_mb = sorted(sample_sizes)[len(sample_sizes) // 2] / 1e6
            if median_size_mb < 10 and workers > 16:
                old_w = workers
                workers = min(workers, 16)
                print(
                    f"  NFS guard: {median_size_mb:.0f} MB median, "
                    f"workers {old_w} -> {workers}"
                )

    print(f"  workers  : {workers}")
    wavs = [Path(p) for p in manifest_df["path"].drop_nulls().to_list()]
    print(f"  files    : {len(wavs)}")

    if not wavs:
        print("Nothing to process.")
        sys.exit(0)

    out_dir = ds.metadata
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume: load previous results
    # ------------------------------------------------------------------
    meta_path = out_dir / "metadata.parquet"
    prev_meta_parts: list[pl.DataFrame] = []
    prev_seg_parts: list[pl.DataFrame] = []

    if meta_path.exists():
        prev_meta_parts.append(pl.read_parquet(meta_path))
    ckpt_meta = out_dir / CKPT_META
    if ckpt_meta.exists():
        prev_meta_parts.append(pl.read_parquet(ckpt_meta))
        print("  Found checkpoint from interrupted run.")

    completed_ids: set[str] = set()
    if prev_meta_parts:
        combined = pl.concat(prev_meta_parts).unique(subset=["file_id"], keep="last")
        completed_ids = set(
            combined.filter(pl.col("success")).get_column("file_id").to_list()
        )

    prev_raw = ds.output / "vad_raw" / "segments.parquet"
    if prev_raw.exists():
        df = pl.read_parquet(prev_raw)
        if "uid" in df.columns:
            df = df.rename({"uid": "file_id"})
        prev_seg_parts.append(df)
    ckpt_segs_p = out_dir / CKPT_SEGS
    if ckpt_segs_p.exists():
        prev_seg_parts.append(pl.read_parquet(ckpt_segs_p))

    if completed_ids:
        before = len(wavs)
        wavs = [w for w in wavs if w.stem not in completed_ids]
        print(f"  Resume: {before - len(wavs)} done, " f"{len(wavs)} remaining")

    # ------------------------------------------------------------------
    # Run VAD
    # ------------------------------------------------------------------
    t0_wall = time.time()
    if wavs:
        meta_rows, seg_rows = run_vad_parallel(
            wavs,
            args.hop_size,
            args.threshold,
            workers,
            checkpoint_dir=out_dir,
        )
    else:
        print("  All files processed. Regenerating outputs.")
        meta_rows, seg_rows = [], []
    wall_seconds = time.time() - t0_wall

    # ------------------------------------------------------------------
    # Merge with previous results
    # ------------------------------------------------------------------
    new_meta_df = pl.DataFrame(meta_rows) if meta_rows else pl.DataFrame()
    new_ids = (
        set(new_meta_df.get_column("file_id").to_list())
        if not new_meta_df.is_empty()
        else set()
    )

    all_meta = [
        df.filter(~pl.col("file_id").is_in(list(new_ids))) for df in prev_meta_parts
    ]
    all_meta = [p for p in all_meta if not p.is_empty()]
    if not new_meta_df.is_empty():
        all_meta.append(new_meta_df)

    if not all_meta:
        print("ERROR: no results.", file=sys.stderr)
        sys.exit(1)

    meta_df = pl.concat(all_meta).sort("path")

    # Segments
    empty_seg_schema = {
        "file_id": pl.Utf8,
        "onset": pl.Float64,
        "offset": pl.Float64,
        "duration": pl.Float64,
    }
    new_seg_df = (
        pl.DataFrame(seg_rows) if seg_rows else pl.DataFrame(schema=empty_seg_schema)
    )
    keep_seg = [
        df.filter(~pl.col("file_id").is_in(list(new_ids))) for df in prev_seg_parts
    ]
    all_seg = [p for p in keep_seg if not p.is_empty()]
    if not new_seg_df.is_empty():
        all_seg.append(new_seg_df)

    seg_df = (
        pl.concat(all_seg).sort("file_id", "onset")
        if all_seg
        else pl.DataFrame(schema=empty_seg_schema)
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    atomic_write_parquet(meta_df, meta_path)

    # Summary
    ok = meta_df.filter(pl.col("success"))
    fail = meta_df.filter(~pl.col("success"))
    print(f"\nSuccess: {len(ok)}/{len(meta_df)}")
    if not fail.is_empty():
        print(f"Failed:  {len(fail)}", file=sys.stderr)
        for row in fail.head(5).iter_rows(named=True):
            name = Path(row["path"]).stem
            print(f"  {name}: {row['error']}", file=sys.stderr)

    # Standardised vad_raw / vad_merged parquets
    vad_raw_df = seg_df.rename({"file_id": "uid"})

    vad_raw_dir = ds.output / "vad_raw"
    vad_raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(vad_raw_df, vad_raw_dir / "segments.parquet")

    vad_merged_df = merge_segments_df(vad_raw_df)
    vad_merged_dir = ds.output / "vad_merged"
    vad_merged_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(vad_merged_df, vad_merged_dir / "segments.parquet")

    print(f"Saved: {len(vad_raw_df):,} raw, " f"{len(vad_merged_df):,} merged segments")

    clear_checkpoint(out_dir)

    # Benchmark
    total_audio_s = meta_df["duration"].sum() if "duration" in meta_df.columns else 0.0
    total_bytes = sum(w.stat().st_size for w in wavs if w.exists())
    log_benchmark(
        step="vad",
        dataset=args.dataset,
        n_files=len(meta_df),
        wall_seconds=wall_seconds,
        total_audio_seconds=total_audio_s,
        total_bytes=total_bytes,
        n_workers=workers,
    )


if __name__ == "__main__":
    main()
