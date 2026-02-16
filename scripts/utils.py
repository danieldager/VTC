import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Dataset path conventions
# ---------------------------------------------------------------------------


@dataclass
class DatasetPaths:
    """Standard paths derived from a dataset name.

    Convention:
        manifests/{dataset}.parquet
        output/{dataset}/
        metadata/{dataset}/
        figures/{dataset}/
    """

    dataset: str
    manifest: Path
    output: Path
    metadata: Path
    figures: Path


def get_dataset_paths(dataset: str) -> DatasetPaths:
    """Derive all standard paths from a dataset name."""
    # Check for .parquet then .csv manifest
    manifest = Path("manifests") / f"{dataset}.parquet"
    if not manifest.exists():
        csv = Path("manifests") / f"{dataset}.csv"
        if csv.exists():
            manifest = csv

    return DatasetPaths(
        dataset=dataset,
        manifest=manifest,
        output=Path("output") / dataset,
        metadata=Path("metadata") / dataset,
        figures=Path("figures") / dataset,
    )


def shard_list(items: list, array_id: int, array_count: int) -> list:
    """Return a contiguous slice of *items* for SLURM array task *array_id*.

    Distributes the remainder across the first shards so sizes differ by at
    most 1 and every shard is contiguous.
    """
    n = len(items)
    base, remainder = divmod(n, array_count)
    if array_id < remainder:
        start = array_id * (base + 1)
        size = base + 1
    else:
        start = remainder * (base + 1) + (array_id - remainder) * base
        size = base
    return items[start : start + size]


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> pl.DataFrame:
    """Read a manifest file (.parquet or .csv) into a Polars DataFrame."""
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def get_task_shard(
    manifest_path: str, array_id: int, array_count: int
) -> tuple[int, int, list[str]]:
    """
    Parses manifest (txt, csv, parquet) and returns (total_files, chunk_size, file_paths)
    for the specific array task.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"{manifest_path} not found")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        # Read all lines
        all_paths = sorted(
            [l.strip() for l in path.read_text().splitlines() if l.strip()]
        )

        total_files = len(all_paths)
        chunk_size = total_files // array_count
        start_idx = array_id * chunk_size
        if array_id == array_count - 1:
            end_idx = total_files
            chunk_size = end_idx - start_idx
        else:
            end_idx = start_idx + chunk_size

        return total_files, chunk_size, all_paths[start_idx:end_idx]

    elif suffix in [".parquet", ".csv"]:
        # Use Polars
        if suffix == ".parquet":
            lf = pl.scan_parquet(manifest_path)
        else:
            lf = pl.scan_csv(manifest_path)

        # Identify path column
        schema = lf.collect_schema()
        if "path" in schema.names():
            col_name = "path"
        elif "audio_filepath" in schema.names():
            col_name = "audio_filepath"
        else:
            raise ValueError(
                f"Manifest {manifest_path} must contain 'path' or 'audio_filepath' column"
            )

        total_files = lf.select(pl.len()).collect().item()
        base_chunk_size = total_files // array_count
        start_idx = array_id * base_chunk_size

        length = base_chunk_size
        if array_id == array_count - 1:
            length = total_files - start_idx

        return (
            total_files,
            length,
            lf.sort(col_name)
            .slice(start_idx, length)
            .select(col_name)
            .collect()
            .get_column(col_name)
            .to_list(),
        )

    else:
        raise ValueError(f"Unsupported manifest extension: {suffix}")


def merge_segments_df(
    df: pl.DataFrame,
    min_duration_off_s: float = 0.1,
    min_duration_on_s: float = 0.1,
) -> pl.DataFrame:
    """Apply collar-based gap filling and minimum duration filter.

    Replicates pyannote's ``Annotation.support(collar)`` followed by
    minimum-duration pruning, implemented purely in Polars.

    Within each partition (``uid``, and ``label`` when present):
      1. Consecutive segments whose gap ≤ ``min_duration_off_s`` are merged.
      2. Resulting segments shorter than ``min_duration_on_s`` are removed.

    Args:
        df: DataFrame with columns ``uid, onset, offset, duration``
            and optionally ``label``.
        min_duration_off_s: Fill gaps ≤ this many seconds (collar).
        min_duration_on_s: Remove segments shorter than this.

    Returns:
        DataFrame with merged/filtered segments and the same column set.
    """
    if df.is_empty():
        return df

    has_label = "label" in df.columns
    partition_cols = ["uid", "label"] if has_label else ["uid"]

    result = (
        df.sort(*partition_cols, "onset")
        # Cumulative max of *previous* offsets within each partition
        .with_columns(
            _prev_max_off=pl.col("offset")
            .shift(1)
            .cum_max()
            .over(*partition_cols)
            .fill_null(0.0)
        )
        # A new group starts when the gap exceeds the collar
        .with_columns(
            _new_grp=(
                pl.col("onset") > (pl.col("_prev_max_off") + min_duration_off_s)
            ).cast(pl.Int32)
        )
        .with_columns(_grp_id=pl.col("_new_grp").cum_sum().over(*partition_cols))
        # Merge each group into a single segment
        .group_by(*partition_cols, "_grp_id")
        .agg(
            onset=pl.col("onset").min(),
            offset=pl.col("offset").max(),
        )
        .with_columns(
            onset=pl.col("onset").round(3),
            offset=pl.col("offset").round(3),
            duration=(pl.col("offset") - pl.col("onset")).round(3),
        )
        # Remove segments shorter than the minimum
        .filter(pl.col("duration") >= min_duration_on_s)
        .drop("_grp_id")
        .sort(*partition_cols, "onset")
    )

    # Ensure consistent column order
    cols = ["uid", "onset", "offset", "duration"]
    if has_label:
        cols.append("label")
    return result.select(cols)


# ---------------------------------------------------------------------------
# Progress logging utilities
# ---------------------------------------------------------------------------


def log_progress(done: int, total: int, t0: float, label: str = "") -> None:
    """Log processing progress with rate and ETA."""
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    eta = f"{remaining / 60:.0f}m" if remaining < 3600 else f"{remaining / 3600:.1f}h"
    prefix = f"{label}: " if label else ""
    print(f"  {prefix}[{done:>7}/{total}]  {rate:.1f} files/s  ETA {eta}")


def get_log_interval(n: int) -> int:
    """Get adaptive reporting interval based on progress."""
    if n < 10_000:
        return 1_000
    if n < 50_000:
        return 5_000
    if n < 100_000:
        return 10_000
    return 20_000


def atomic_write_parquet(
    df: pl.DataFrame, path: Path | str, compression: str = "zstd"
) -> None:
    """Write a Parquet file atomically via temp-file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression=compression)  # type: ignore
    tmp.rename(path)
