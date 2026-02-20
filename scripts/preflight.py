#!/usr/bin/env python3
"""
Preflight check: summarise the dataset before pipeline execution.

Reports:
  - Number of audio files
  - Total size on disk (with human-readable units)
  - Estimated wall-clock time for the full pipeline

Usage:
    python scripts/preflight.py chunks30
    python scripts/preflight.py chunks30 --sample 500
"""

import argparse
import os
import sys
from pathlib import Path

from scripts.utils import (
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight check — summarise the dataset and estimate pipeline "
            "duration before submitting any jobs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/preflight.py chunks30\n"
            "  python scripts/preflight.py chunks30 --sample 500\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name — used to derive output directories.",
    )
    parser.add_argument(
        "--vtc-tasks",
        type=int,
        default=4,
        help="Number of parallel VTC GPU tasks (default: 4). Affects ETA.",
    )
    parser.add_argument(
        "--vad-workers",
        type=int,
        default=48,
        help="Number of parallel VAD CPU workers (default: 48). Affects ETA.",
    )
    add_sample_argument(parser)
    args = parser.parse_args()

    # ---- Resolve manifest ----
    ds = get_dataset_paths(args.dataset)
    df = load_manifest(ds.manifest)

    # ---- Apply sampling if requested ----
    df = sample_manifest(df, args.sample)

    # ---- Resolve audio paths ----
    resolved = df["path"].drop_nulls().to_list()

    # ---- Stat the files ----
    stats = stat_files(resolved)

    # ---- Estimate durations ----
    total_bytes = stats["total_bytes"]
    bench_vad, bench_vtc = _rates_from_benchmarks()
    using_benchmarks = bench_vad is not None or bench_vtc is not None

    # VAD: stored benchmark rate is bytes/s per-worker, scale by actual workers
    vad_per_worker = bench_vad if bench_vad is not None else _DEFAULT_VAD_BYTES_PER_S
    scaled_vad_rate = vad_per_worker * args.vad_workers
    t_vad = total_bytes / scaled_vad_rate if scaled_vad_rate > 0 else 0

    # VTC: stored benchmark rate is bytes/s per-GPU, scale by GPU tasks
    vtc_per_gpu = bench_vtc if bench_vtc is not None else _DEFAULT_VTC_BYTES_PER_S
    t_vtc = total_bytes / (vtc_per_gpu * args.vtc_tasks) if args.vtc_tasks > 0 else 0

    t_total = t_vad + t_vtc + COMPARE_FIXED_SEC

    # ---- Print report ----
    print()
    print("Preflight")
    print("━" * 50)
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
    print()
    print("  Estimated duration:")
    vad_str = human_duration(t_vad)
    vtc_str = human_duration(t_vtc)
    cmp_str = human_duration(COMPARE_FIXED_SEC)
    tot_str = human_duration(t_total)
    print(f"    VAD  ({args.vad_workers} workers) : {vad_str}")
    print(f"    VTC  ({args.vtc_tasks} GPUs)    : {vtc_str}")
    print(f"    Compare          : {cmp_str}")
    print(f"    {'─' * 28}")
    print(f"    Total            : {tot_str}")
    print()
    if using_benchmarks:
        src_vad = "benchmark" if bench_vad else "heuristic"
        src_vtc = "benchmark" if bench_vtc else "heuristic"
        print(
            f"  Rates: VAD={src_vad}, VTC={src_vtc}"
        )
    else:
        print(
            "  Rates: heuristic estimates "
            "(run jobs to calibrate)"
        )
    print("━" * 50)
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
