"""
Analyze VAD coverage of VTC segments.

Computes:
- Number of VTC segments that are cut by VAD (i.e., not fully covered)
- Total duration of speech missed by VAD (i.e., cut VTC segments)

Broken down by speaker type (label) and globally.
"""

import logging
from bisect import bisect_left
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


def load_segments(path: Path) -> pl.DataFrame:
    """Load and concatenate parquet files from a directory."""
    parquet_files = sorted(path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {path}")

    dfs = [pl.read_parquet(f) for f in parquet_files]
    return pl.concat(dfs, how="vertical")


def count_overlaps(intervals: list[tuple[float, float]]) -> int:
    """
    Count the number of overlapping pairs in a sorted list of intervals.
    Used to sanity-check that the VAD segments are non-overlapping after merging.
    """
    count = 0
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:
            count += 1
    return count


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Merge overlapping/touching intervals and return the sorted onset list for binary search.

    Args:
        intervals: List of (onset, offset) tuples (unsorted).

    Returns:
        (merged_intervals, onsets) where onsets is the precomputed list of start times.
    """
    if not intervals:
        return [], []

    intervals = sorted(intervals)
    merged = [intervals[0]]

    for onset, offset in intervals[1:]:
        last_onset, last_offset = merged[-1]
        # Merge if overlapping or touching (within 1ms tolerance for floating point)
        if onset <= last_offset + 1e-3:
            merged[-1] = (last_onset, max(last_offset, offset))
        else:
            merged.append((onset, offset))

    onsets = [v[0] for v in merged]
    return merged, onsets


def compute_cuts_and_missed(
    vtc_onset: float,
    vtc_offset: float,
    vad_intervals: list[tuple[float, float]],
    vad_onsets: list[float],
) -> tuple[int, float]:
    """
    Compute the number of cuts and missed duration for a single VTC segment.

    A "cut" is each contiguous region within the VTC segment that is NOT covered
    by any VAD interval. For example:

        VAD:  [-----]    [------]
        VTC:      [----------]

    This produces one cut (the gap between the two VAD intervals).

    Args:
        vtc_onset:     Start time of the VTC segment.
        vtc_offset:    End time of the VTC segment.
        vad_intervals: Sorted, non-overlapping list of (onset, offset) VAD intervals.
        vad_onsets:    Precomputed list of VAD onset times (vad_intervals[i][0]).

    Returns:
        (n_cuts, missed_duration)
    """
    if not vad_intervals:
        return 1, vtc_offset - vtc_onset

    # Find the first VAD interval whose onset is < vtc_offset (i.e. could overlap)
    # bisect_left gives us the first index with onset >= vtc_onset;
    # step back one to catch intervals that started before vtc_onset but may still overlap.
    start_idx = bisect_left(vad_onsets, vtc_onset)
    if start_idx > 0:
        start_idx -= 1

    n_cuts = 0
    missed = 0.0
    pos = vtc_onset  # current position as we walk through the VTC segment

    for i in range(start_idx, len(vad_intervals)):
        vad_onset, vad_offset = vad_intervals[i]

        if vad_onset >= vtc_offset:
            break  # all remaining VAD intervals are past the VTC segment

        if vad_offset <= vtc_onset:
            continue  # this VAD interval ends before the VTC segment starts

        overlap_start = max(vad_onset, vtc_onset)
        overlap_end = min(vad_offset, vtc_offset)

        if overlap_start > pos:
            # Gap between current position and where VAD coverage starts
            n_cuts += 1
            missed += overlap_start - pos

        pos = max(pos, overlap_end)

    # Trailing region after all VAD intervals
    if pos < vtc_offset:
        n_cuts += 1
        missed += vtc_offset - pos

    return n_cuts, missed


def analyze_vad_coverage(vad_path: Path, vtc_path: Path) -> dict[str, Any]:
    """
    Analyze VAD coverage of VTC segments.

    Args:
        vad_path: Path to VAD merged parquet directory.
        vtc_path: Path to VTC merged parquet directory.

    Returns:
        Dictionary with statistics.
    """
    logger.info("Loading VAD segments...")
    vad = load_segments(vad_path)

    logger.info("Loading VTC segments...")
    vtc = load_segments(vtc_path)

    # Build a lookup of merged VAD intervals (+ precomputed onsets) by uid
    logger.info("Building VAD intervals lookup by file...")
    total_vad_overlaps = 0
    vad_by_uid: dict[str, tuple[list[tuple[float, float]], list[float]]] = {}
    for uid in vad["uid"].unique().to_list():
        uid_segments = vad.filter(pl.col("uid") == uid).sort("onset")
        raw_intervals = [
            (row["onset"], row["offset"])
            for row in uid_segments.select(["onset", "offset"]).iter_rows(named=True)
        ]
        # Log overlaps in the raw (pre-merge) data
        total_vad_overlaps += count_overlaps(raw_intervals)
        merged, onsets = merge_intervals(raw_intervals)
        vad_by_uid[uid] = (merged, onsets)

    if total_vad_overlaps > 0:
        logger.warning(
            f"Found {total_vad_overlaps} overlapping VAD segment pairs across all files (merged away)."
        )
    else:
        logger.info("No overlapping VAD segments found — data is clean.")

    # Analyze each VTC segment
    logger.info("Analyzing VTC segments...")
    stats: dict[str, Any] = {
        "total_vtc_segments": 0,
        "total_vtc_duration": 0.0,
        "total_cuts": 0,
        "cut_vtc_segments": 0,
        "missed_duration": 0.0,
        "by_label": {},
    }

    # Initialize by-label counters
    for label in ["KCHI", "OCH", "MAL", "FEM"]:
        stats["by_label"][label] = {
            "total_segments": 0,
            "total_duration": 0.0,
            "total_cuts": 0,
            "cut_segments": 0,
            "missed_duration": 0.0,
        }

    for row in vtc.iter_rows(named=True):
        uid = row["uid"]
        onset = row["onset"]
        offset = row["offset"]
        label = row["label"]

        stats["total_vtc_segments"] += 1
        stats["total_vtc_duration"] += offset - onset
        stats["by_label"][label]["total_segments"] += 1
        stats["by_label"][label]["total_duration"] += offset - onset

        vad_intervals, vad_onsets = vad_by_uid.get(uid, ([], []))
        n_cuts, missed = compute_cuts_and_missed(
            onset, offset, vad_intervals, vad_onsets
        )

        if n_cuts > 0:
            stats["total_cuts"] += n_cuts
            stats["cut_vtc_segments"] += 1
            stats["missed_duration"] += missed
            stats["by_label"][label]["total_cuts"] += n_cuts
            stats["by_label"][label]["cut_segments"] += 1
            stats["by_label"][label]["missed_duration"] += missed

    return stats


def format_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Format statistics for output."""
    formatted: dict[str, list[str]] = {
        "metric": [],
        "all_types": [],
        "KCHI": [],
        "OCH": [],
        "MAL": [],
        "FEM": [],
    }

    total_segments = stats["total_vtc_segments"]
    total_duration = stats["total_vtc_duration"]
    cut_segments = stats["cut_vtc_segments"]
    total_cuts = stats["total_cuts"]
    missed_duration = stats["missed_duration"]
    cut_pct = (cut_segments / total_segments * 100) if total_segments > 0 else 0
    missed_pct = (missed_duration / total_duration * 100) if total_duration > 0 else 0

    rows: list[tuple[str, str, str]] = [
        # (metric name, all_types value, per-label key)
        ("Total VTC segments", str(total_segments), "total_segments"),
        ("Total VTC duration (seconds)", f"{total_duration:.3f}", "total_duration"),
        ("VTC segments with >=1 cut", str(cut_segments), "cut_segments"),
        ("% VTC segments with >=1 cut", f"{cut_pct:.2f}%", "percent_cut"),
        ("Total number of cuts", str(total_cuts), "total_cuts"),
        ("Total missed speech (seconds)", f"{missed_duration:.3f}", "missed_duration"),
        ("% VTC speech missed", f"{missed_pct:.2f}%", "percent_missed"),
    ]

    for metric, all_val, label_key in rows:
        formatted["metric"].append(metric)
        formatted["all_types"].append(all_val)
        for label in ["KCHI", "OCH", "MAL", "FEM"]:
            ls = stats["by_label"][label]
            if label_key == "percent_cut":
                total = ls["total_segments"]
                cut = ls["cut_segments"]
                val = f"{(cut / total * 100) if total > 0 else 0:.2f}%"
            elif label_key == "percent_missed":
                tot_dur = ls["total_duration"]
                val = f"{(ls['missed_duration'] / tot_dur * 100) if tot_dur > 0 else 0:.2f}%"
            elif label_key in ("missed_duration", "total_duration"):
                val = f"{ls[label_key]:.3f}"
            else:
                val = str(ls[label_key])
            formatted[label].append(val)

    return formatted


def main(dataset: str = "seedlings_10", output_dir: Path | None = None):
    """
    Main entry point for VAD coverage analysis.

    Args:
        dataset: Dataset name (used to locate segments).
        output_dir: Output directory for CSV. Defaults to project output/{dataset}.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set paths
    project_root = Path(__file__).parent.parent.parent
    vad_path = project_root / "output" / dataset / "vad_merged"
    vtc_path = project_root / "output" / dataset / "vtc_merged"

    if not vad_path.exists():
        raise FileNotFoundError(f"VAD path not found: {vad_path}")
    if not vtc_path.exists():
        raise FileNotFoundError(f"VTC path not found: {vtc_path}")

    # Analyze coverage
    stats = analyze_vad_coverage(vad_path, vtc_path)
    formatted = format_stats(stats)

    # Log results
    logger.info("\n" + "=" * 80)
    logger.info("VAD COVERAGE ANALYSIS RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nTotal VTC segments:    {stats['total_vtc_segments']}")
    logger.info(f"Segments with >=1 cut: {stats['cut_vtc_segments']}")
    logger.info(f"Total cuts:            {stats['total_cuts']}")
    logger.info(f"Total missed duration: {stats['missed_duration']:.3f}s")
    logger.info("\nBy speaker type:")
    for label in ["KCHI", "OCH", "MAL", "FEM"]:
        ls = stats["by_label"][label]
        pct = (
            ls["cut_segments"] / ls["total_segments"] * 100
            if ls["total_segments"] > 0
            else 0
        )
        logger.info(
            f"  {label}: {ls['cut_segments']}/{ls['total_segments']} segs cut ({pct:.1f}%) | "
            f"{ls['total_cuts']} cuts | {ls['missed_duration']:.3f}s missed"
        )
    logger.info("=" * 80 + "\n")

    # Write CSV
    if output_dir is None:
        output_dir = project_root / "output" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "vad_coverage_analysis.csv"

    # Convert to DataFrame and write
    df = pl.DataFrame(formatted)
    df.write_csv(str(output_csv))
    logger.info(f"Results written to: {output_csv}")

    return stats


if __name__ == "__main__":
    main()
