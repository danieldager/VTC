import argparse
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Supported manifest extensions (in priority order for auto-detection)
# ---------------------------------------------------------------------------

SUPPORTED_MANIFEST_EXTENSIONS: list[str] = [
    ".parquet",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".json",
    ".jsonl",
]


# ---------------------------------------------------------------------------
# Dataset path conventions
# ---------------------------------------------------------------------------


@dataclass
class DatasetPaths:
    """Standard paths derived from a dataset name.

    Convention:
        manifests/{dataset}.<ext>
        output/{dataset}/
        metadata/{dataset}/
        figures/{dataset}/
    """

    dataset: str
    manifest: Path
    output: Path
    metadata: Path
    figures: Path


def resolve_manifest(manifest_arg: str) -> Path:
    """Resolve a manifest argument to a concrete file path.

    The argument can be:
      1. A path to an existing file (absolute or relative), e.g.
         ``/data/my_manifest.csv`` or ``manifests/chunks30.parquet``.
      2. A bare dataset name (no path separators, no recognised extension),
         e.g. ``chunks30``.  In this case the function searches
         ``manifests/`` for files whose stem matches.

    Raises:
        FileNotFoundError: if no matching manifest exists.
        ValueError:        if a bare name matches multiple files and the
                           intended extension is ambiguous.
    """
    p = Path(manifest_arg)

    # --- Case 1: looks like an explicit path ---
    if (
        "/" in manifest_arg
        or "\\" in manifest_arg
        or p.suffix.lower() in SUPPORTED_MANIFEST_EXTENSIONS
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {p.resolve()}\n"
                f"Supported formats: {', '.join(SUPPORTED_MANIFEST_EXTENSIONS)}"
            )
        if p.suffix.lower() not in SUPPORTED_MANIFEST_EXTENSIONS:
            raise ValueError(
                f"Unsupported manifest format '{p.suffix}'. "
                f"Supported: {', '.join(SUPPORTED_MANIFEST_EXTENSIONS)}"
            )
        return p

    # --- Case 2: bare dataset name → search manifests/ ---
    manifests_dir = Path("manifests")
    if not manifests_dir.is_dir():
        raise FileNotFoundError(
            f"No 'manifests/' directory found and '{manifest_arg}' is not a "
            f"path to an existing file."
        )

    matches = [
        f
        for ext in SUPPORTED_MANIFEST_EXTENSIONS
        for f in manifests_dir.glob(f"{manifest_arg}{ext}")
        if f.is_file()
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No manifest found for dataset '{manifest_arg}'.\n"
            f"Searched: manifests/{manifest_arg}.<ext> with ext in "
            f"{SUPPORTED_MANIFEST_EXTENSIONS}"
        )

    if len(matches) == 1:
        return matches[0]

    # Multiple matches — prefer .csv (the standard normalised format)
    csv_matches = [m for m in matches if m.suffix.lower() == ".csv"]
    if len(csv_matches) == 1:
        return csv_matches[0]

    # Still ambiguous
    found_list = ", ".join(str(m) for m in matches)
    raise ValueError(
        f"Ambiguous: multiple manifests match dataset name '{manifest_arg}':\n"
        f"  {found_list}\n"
        f"Please specify the full path or include the extension, e.g. "
        f"'{matches[0]}'."
    )


def get_dataset_paths(dataset: str) -> DatasetPaths:
    """Derive all standard paths from a dataset name.

    The manifest is auto-detected from ``manifests/{dataset}.<ext>``.
    When multiple formats exist, ``.csv`` is preferred (the standard
    normalised format produced by ``normalize.py``).
    """
    return DatasetPaths(
        dataset=dataset,
        manifest=resolve_manifest(dataset),
        output=Path("output") / dataset,
        metadata=Path("metadata") / dataset,
        figures=Path("figures") / dataset,
    )


def validate_path_column(df: pl.DataFrame, path_col: str, manifest_path: Path) -> None:
    """Assert that *path_col* exists in *df* and contains non-null strings.

    Raises:
        SystemExit with a helpful message listing available columns.
    """
    if path_col not in df.columns:
        available = ", ".join(f"'{c}'" for c in df.columns)
        print(
            f"ERROR: Column '{path_col}' not found in manifest "
            f"{manifest_path}.\n"
            f"  Available columns: {available}\n"
            f"  Use --path-col to specify the correct column name.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_null = df[path_col].null_count()
    if n_null > 0:
        print(
            f"WARNING: {n_null} null value(s) in column '{path_col}'; "
            f"those rows will be skipped.",
            file=sys.stderr,
        )


def resolve_audio_paths(
    paths: list[str],
    audio_root: str | None = None,
) -> list[str]:
    """Prepend *audio_root* to relative paths.

    If *audio_root* is ``None`` paths are returned as-is.  Absolute paths
    are never modified.
    """
    if audio_root is None:
        return paths
    root = Path(audio_root)
    if not root.is_dir():
        print(
            f"ERROR: --audio-root directory does not exist: {root.resolve()}",
            file=sys.stderr,
        )
        sys.exit(1)
    resolved = []
    for p in paths:
        pp = Path(p)
        resolved.append(str(root / pp) if not pp.is_absolute() else p)
    return resolved


def sample_manifest(
    df: pl.DataFrame,
    sample: int | float | None,
    seed: int = 42,
) -> pl.DataFrame:
    """Optionally sub-sample a manifest DataFrame.

    Args:
        df:     The full manifest DataFrame.
        sample: How many rows to keep.  Interpretation:
                  * ``None`` → no sampling, return *df* unchanged.
                  * ``int`` (≥ 1) → keep exactly that many rows (random).
                  * ``float`` in (0, 1) → keep that fraction of rows.
                Rows are sampled randomly with a fixed seed for
                reproducibility.
        seed:   Random seed for reproducible sampling.

    Returns:
        A (possibly smaller) DataFrame.
    """
    if sample is None:
        return df

    n_total = len(df)

    if isinstance(sample, float) and 0 < sample < 1:
        n_keep = max(1, int(round(n_total * sample)))
    elif isinstance(sample, (int, float)) and sample >= 1:
        n_keep = int(sample)
    else:
        raise ValueError(
            f"Invalid --sample value: {sample!r}.  "
            f"Use an integer ≥ 1 (number of files) or a float in (0, 1) "
            f"(fraction of files)."
        )

    if n_keep >= n_total:
        return df

    return df.sample(n=n_keep, seed=seed)


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
        start = (
            remainder * (base + 1) + (array_id - remainder) * base
        )
        size = base
    return items[start : start + size]


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> pl.DataFrame:
    """Read a manifest file into a Polars DataFrame.

    Supported formats: .parquet, .csv, .tsv, .xlsx, .xls, .json, .jsonl
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".tsv":
        return pl.read_csv(path, separator="\t")
    if suffix in (".xlsx", ".xls"):
        try:
            return pl.read_excel(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read Excel file '{path}'. Make sure 'openpyxl' "
                f"(for .xlsx) or 'xlrd' (for .xls) is installed.\n  Error: {e}"
            ) from e
    if suffix == ".json":
        return pl.read_json(path)
    if suffix == ".jsonl":
        return pl.read_ndjson(path)
    raise ValueError(
        f"Unsupported manifest format '{suffix}'.  "
        f"Supported: {', '.join(SUPPORTED_MANIFEST_EXTENSIONS)}"
    )


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


def atomic_write_parquet(
    df: pl.DataFrame, path: Path | str, compression: str = "zstd"
) -> None:
    """Write a Parquet file atomically via temp-file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression=compression)  # type: ignore
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Pipeline benchmark logging
# ---------------------------------------------------------------------------

BENCHMARK_LOG = Path("logs/benchmarks.jsonl")


def _get_gpu_name() -> str:
    """Best-effort GPU name (empty string if unavailable)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def _hardware_info(n_workers: int | None = None) -> dict:
    """Collect hardware context for a benchmark record."""
    return {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count() or 0,
        "n_workers": n_workers,
        "gpu": _get_gpu_name(),
        "platform": platform.platform(),
    }


def log_benchmark(
    step: str,
    dataset: str,
    n_files: int,
    wall_seconds: float,
    total_audio_seconds: float = 0.0,
    total_bytes: int = 0,
    n_workers: int | None = None,
    extra: dict | None = None,
) -> None:
    """Append a benchmark record to ``logs/benchmarks.jsonl``.

    Each record captures enough context to build predictive ETA models:
    step name, dataset size in files/bytes/audio-duration, wall-time,
    and hardware information.
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "step": step,
        "dataset": dataset,
        "n_files": n_files,
        "wall_seconds": round(wall_seconds, 2),
        "total_audio_seconds": round(total_audio_seconds, 2),
        "total_bytes": total_bytes,
        "files_per_second": round(n_files / wall_seconds, 2) if wall_seconds > 0 else 0,
        "bytes_per_second": round(total_bytes / wall_seconds, 2) if wall_seconds > 0 and total_bytes > 0 else 0,
        "hardware": _hardware_info(n_workers),
    }
    if extra:
        record["extra"] = extra

    BENCHMARK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_benchmarks(step: str | None = None) -> list[dict]:
    """Read benchmark records, optionally filtered by *step*."""
    if not BENCHMARK_LOG.exists():
        return []
    records = []
    for line in BENCHMARK_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if step is None or rec.get("step") == step:
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def add_sample_argument(parser) -> None:  # noqa: ANN001 (argparse.ArgumentParser)
    """Add ``--sample`` to an argparse parser.

    This is the only manifest-related argument downstream scripts need;
    ``--manifest``, ``--path-col``, and ``--audio-root`` are handled by the
    normalization step (``normalize.py``) before any processing begins.
    """
    parser.add_argument(
        "--sample",
        default=None,
        type=_parse_sample,
        metavar="N_OR_FRAC",
        help=(
            "Process only a random subset of the dataset (for testing).  "
            "Pass an integer ≥ 1 to select that many files, or a float in "
            "(0, 1) to select that fraction.  E.g. --sample 500 keeps 500 "
            "files; --sample 0.1 keeps 10%%.  Sampling is deterministic "
            "(seed=42) so repeated runs select the same subset."
        ),
    )


def _parse_sample(value: str) -> int | float:
    """Parse --sample value as int or float."""
    try:
        if "." in value:
            f = float(value)
            if 0 < f < 1:
                return f
            raise ValueError
        n = int(value)
        if n < 1:
            raise ValueError
        return n
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid sample value '{value}'.  "
            f"Use an integer ≥ 1 (number of files) or a float in (0, 1) "
            f"(fraction of files).  E.g. --sample 500 or --sample 0.1"
        )
