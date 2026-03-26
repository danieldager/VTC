#!/usr/bin/env python3
"""
Preflight check: summarise the dataset before pipeline execution.

Reports:
  - Number of audio files
  - Total size on disk (with human-readable units)
  - GPU detection and recommended batch sizes
  - Array-count allocation for SLURM jobs
  - Estimated wall-clock time for the full pipeline

The ``--emit-env`` flag prints machine-readable KEY=VALUE lines (one per
line) that ``pipeline.sh`` can ``eval`` to configure its sbatch calls.

Usage:
    python -m src.pipeline.preflight chunks30
    python -m src.pipeline.preflight chunks30 --sample 500
    eval "$(python -m src.pipeline.preflight chunks30 --emit-env)"
"""

import argparse
import os
import sys
from pathlib import Path

from src.pipeline.resources import (
    gather_dataset_stats,
    plan_resources,
    query_partition_gpus,
)
from src.utils import (
    add_sample_argument,
    get_dataset_paths,
    load_benchmarks,
    load_manifest,
    sample_manifest,
)


# ---------------------------------------------------------------------------
# Processing-rate estimates
# ---------------------------------------------------------------------------
# Fallback heuristic rates (bytes / second) — used when no benchmark data exists.
#   VAD  (CPU, multiprocessed):  ~820 MB/s  with 32 workers  (~25.6 MB/s per worker)
#   VTC  (GPU, single process):  ~50 MB/s   per GPU
#   Compare (CPU):               very fast, dominated by I/O

_DEFAULT_VAD_BYTES_PER_S = 25.6e6   # bytes/s per worker (heuristic)
_DEFAULT_VTC_BYTES_PER_S = 50.0e6   # bytes/s per GPU    (heuristic)
COMPARE_FIXED_SEC = 30.0


def _rates_from_benchmarks() -> tuple[float | None, float | None]:
    """Derive processing rates from historical benchmark data.

    Returns (vad_rate, vtc_rate) in **bytes per second** — either may be
    ``None`` if no usable data exists.

    VAD rates are normalised to bytes/s per worker; VTC rates are per-GPU.
    """
    vad_rate = vtc_rate = None

    vad_records = load_benchmarks("vad")
    if vad_records:
        rates = []
        for r in vad_records:
            nw = (r.get("hardware") or {}).get("n_workers") or 1
            bps = r.get("bytes_per_second", 0)
            if bps > 0:
                rates.append(bps / nw)  # per-worker rate
        if rates:
            rates.sort()
            vad_rate = rates[len(rates) // 2]  # median

    vtc_records = load_benchmarks("vtc")
    if vtc_records:
        rates = [r["bytes_per_second"] for r in vtc_records if r.get("bytes_per_second", 0) > 0]
        if rates:
            rates.sort()
            vtc_rate = rates[len(rates) // 2]

    return vad_rate, vtc_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def human_size(nbytes: int | float) -> str:
    """Format a byte count as a human-readable string (e.g. '3.2 GB')."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


def human_duration(seconds: float) -> str:
    """Format seconds into a readable string like '2h 15m' or '45s'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    h, remainder = divmod(seconds, 3600)
    m, _ = divmod(remainder, 60)
    return f"{int(h)}h {int(m)}m"


def stat_files(paths: list[str]) -> dict:
    """Gather file-level statistics.

    Returns dict with keys:
        total, found, missing, total_bytes, missing_paths (first 10)
    """
    total_bytes = 0
    missing: list[str] = []
    found = 0

    for p in paths:
        try:
            total_bytes += os.path.getsize(p)
            found += 1
        except OSError:
            missing.append(p)

    return {
        "total": len(paths),
        "found": found,
        "missing": len(missing),
        "total_bytes": total_bytes,
        "missing_paths": missing[:10],
    }


# Default SLURM partitions to query for GPU info
_DEFAULT_PARTITIONS = ["erc-dupoux", "gpu-p2", "gpu-p1"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight check — summarise the dataset, detect GPU resources, "
            "and recommend batch sizes / array counts before submitting jobs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.pipeline.preflight chunks30\n"
            "  python -m src.pipeline.preflight chunks30 --sample 500\n"
            "  eval \"$(python -m src.pipeline.preflight chunks30 --emit-env)\"\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name — used to derive output directories.",
    )
    parser.add_argument(
        "--vtc-tasks",
        type=int,
        default=None,
        help="Override VTC GPU array count (default: auto-detect).",
    )
    parser.add_argument(
        "--vad-workers",
        type=int,
        default=48,
        help="Number of parallel VAD CPU workers (default: 48). Affects ETA.",
    )
    parser.add_argument(
        "--partitions",
        default=",".join(_DEFAULT_PARTITIONS),
        help=(
            "Comma-separated SLURM partitions to check for GPU info "
            f"(default: {','.join(_DEFAULT_PARTITIONS)})"
        ),
    )
    parser.add_argument(
        "--emit-env",
        action="store_true",
        help=(
            "Print only KEY=VALUE lines suitable for eval in shell scripts. "
            "Suppresses the human-readable report."
        ),
    )
    add_sample_argument(parser)
    args = parser.parse_args()

    # ---- Resolve manifest ----
    ds = get_dataset_paths(args.dataset)
    df = load_manifest(ds.manifest)
    df = sample_manifest(df, args.sample)
    resolved = df["path"].drop_nulls().to_list()

    # ---- Dataset statistics ----
    dstats = gather_dataset_stats(resolved)

    # ---- Backward-compat stat_files dict ----
    stats = stat_files(resolved)

    # ---- GPU detection ----
    partitions = [p.strip() for p in args.partitions.split(",")]
    gpu = query_partition_gpus(partitions)

    # ---- Resource plan ----
    plan = plan_resources(
        dstats,
        gpu,
        max_vtc_shards=args.vtc_tasks,
    )

    # ---- Emit machine-readable env and exit ----
    if args.emit_env:
        print(f"VTC_BATCH_SIZE={plan.vtc_batch_size}")
        print(f"VTC_ARRAY_COUNT={plan.vtc_array_count}")
        print(f"SNR_ARRAY_COUNT={plan.snr_array_count}")
        print(f"NOISE_ARRAY_COUNT={plan.noise_array_count}")
        print(f"GPU_NAME={plan.gpu_name}")
        print(f"GPU_VRAM_GB={plan.gpu_vram_gb}")
        print(f"DATASET_FILES={dstats.n_found}")
        print(f"DATASET_BYTES={dstats.total_bytes}")
        return

    # ---- Estimate durations ----
    total_bytes = stats["total_bytes"]
    bench_vad, bench_vtc = _rates_from_benchmarks()
    using_benchmarks = bench_vad is not None or bench_vtc is not None

    vad_per_worker = bench_vad if bench_vad is not None else _DEFAULT_VAD_BYTES_PER_S
    scaled_vad_rate = vad_per_worker * args.vad_workers
    t_vad = total_bytes / scaled_vad_rate if scaled_vad_rate > 0 else 0

    vtc_per_gpu = bench_vtc if bench_vtc is not None else _DEFAULT_VTC_BYTES_PER_S
    vtc_tasks = plan.vtc_array_count
    t_vtc = total_bytes / (vtc_per_gpu * vtc_tasks) if vtc_tasks > 0 else 0

    t_total = t_vad + t_vtc + COMPARE_FIXED_SEC

    # ---- Print report ----
    print()
    print("Preflight")
    print("━" * 60)
    print(f"  Dataset  : {args.dataset}")
    print(f"  Manifest : {ds.manifest}")
    if args.sample is not None:
        print(f"  Sample   : {args.sample}")
    print()
    print(f"  Files    : {stats['total']:,}")
    print(f"  On disk  : {stats['found']:,}")
    if stats["missing"] > 0:
        print(f"  Missing  : {stats['missing']:,}")
        for p in stats["missing_paths"]:
            name = Path(p).name
            print(f"    {name}")
        if stats["missing"] > 10:
            print(f"    ... and {stats['missing'] - 10} more")
    print(f"  Size     : {human_size(stats['total_bytes'])}")
    if dstats.n_found > 0:
        print(
            f"  Per file : min={human_size(dstats.min_bytes)}  "
            f"mean={human_size(dstats.mean_bytes)}  "
            f"median={human_size(dstats.median_bytes)}  "
            f"max={human_size(dstats.max_bytes)}"
        )

    # ---- GPU & resource plan ----
    print()
    print("  GPU resources:")
    if gpu:
        print(f"    Node GPU   : {gpu.name} ({gpu.vram_gb} GB) × {gpu.count}")
    else:
        print("    Node GPU   : not detected (will use defaults)")
    print(f"    VTC batch  : {plan.vtc_batch_size}")
    print(f"    VTC shards : {plan.vtc_array_count}")
    print(f"    SNR shards : {plan.snr_array_count}")
    print(f"    Noise shards: {plan.noise_array_count}")
    for note in plan.notes:
        print(f"    ▸ {note}")

    # ---- ETA ----
    print()
    print("  Estimated duration:")
    vad_str = human_duration(t_vad)
    vtc_str = human_duration(t_vtc)
    cmp_str = human_duration(COMPARE_FIXED_SEC)
    tot_str = human_duration(t_total)
    print(f"    VAD  ({args.vad_workers} workers) : {vad_str}")
    print(f"    VTC  ({vtc_tasks} GPUs)    : {vtc_str}")
    print(f"    Compare          : {cmp_str}")
    print(f"    {'─' * 28}")
    print(f"    Total            : {tot_str}")
    print()
    if using_benchmarks:
        src_vad = "benchmark" if bench_vad else "heuristic"
        src_vtc = "benchmark" if bench_vtc else "heuristic"
        print(f"  Rates: VAD={src_vad}, VTC={src_vtc}")
    else:
        print("  Rates: heuristic estimates (run jobs to calibrate)")
    print("━" * 60)
    print()

    # Exit with error if no files found
    if stats["found"] == 0:
        print(
            "ERROR: No audio files found on disk.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
