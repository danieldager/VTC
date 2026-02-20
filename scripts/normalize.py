#!/usr/bin/env python3
"""
Normalize a manifest into a standardized CSV.

Creates ``manifests/{dataset}.csv`` with:
  - A ``path`` column containing **absolute** audio file paths
  - All other columns from the source manifest preserved
  - If the source already had a ``path`` column with relative paths,
    the original is kept as ``path_legacy``

After normalization every downstream script can assume:
  - Manifest lives at ``manifests/{dataset}.csv``
  - Audio paths are in the ``path`` column and are absolute

Usage (typically called automatically from pipeline.sh):
    python scripts/normalize.py my_data --manifest /data/meta.xlsx \\
        --path-col recording_id --audio-root /store/audio/
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from scripts.utils import (
    load_manifest,
    resolve_audio_paths,
    resolve_manifest,
    validate_path_column,
)


def normalize_manifest(
    dataset: str,
    manifest: str | None = None,
    path_col: str = "path",
    audio_root: str | None = None,
) -> Path:
    """Create a standardized manifest at ``manifests/{dataset}.csv``.

    Args:
        dataset:    Dataset name — determines the output filename.
        manifest:   Path (or bare name) of the source manifest.
                    Falls back to auto-detecting from dataset name.
        path_col:   Column in the source manifest that holds audio paths.
        audio_root: Optional root directory prepended to relative paths.

    Returns:
        Path to the newly created ``manifests/{dataset}.csv``.
    """
    source_arg = manifest if manifest else dataset
    source = resolve_manifest(source_arg)
    target = Path("manifests") / f"{dataset}.csv"

    print(f"  Source    : {source}")
    print(f"  Target    : {target}")
    print(f"  Path col  : {path_col}")
    if audio_root:
        print(f"  Audio root: {audio_root}")

    # Load source
    df = load_manifest(source)
    validate_path_column(df, path_col, source)

    # Drop rows with null paths
    n_before = len(df)
    df = df.filter(pl.col(path_col).is_not_null())
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with null paths")

    # Resolve paths to absolute
    raw_paths: list[str] = df[path_col].to_list()
    resolved: list[str] = resolve_audio_paths(raw_paths, audio_root)

    # Build the standardized 'path' column
    if path_col == "path":
        # Check if any paths actually changed (i.e. were relative)
        changed = any(r != o for r, o in zip(resolved, raw_paths))
        if changed:
            df = df.rename({"path": "path_legacy"})
            df = df.with_columns(pl.Series("path", resolved))
        else:
            # Already absolute — update in place (no-op in value, but
            # guarantees the column exists after all code paths)
            df = df.with_columns(pl.Series("path", resolved))
    else:
        # Source used a different column name — add 'path', keep original
        df = df.with_columns(pl.Series("path", resolved))

    # Ensure 'path' is the first column
    cols = ["path"] + [c for c in df.columns if c != "path"]
    df = df.select(cols)

    # Warn if overwriting
    if target.exists():
        print(f"  Overwriting: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(target)
    print(
        f"  Normalized: {target}  "
        f"({len(df)} rows, {len(df.columns)} cols)"
    )

    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize a manifest into a standardized CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/normalize.py my_data --manifest /data/meta.xlsx "
            "--path-col recording_id\n"
            "  python scripts/normalize.py my_data --manifest metadata.csv "
            "--audio-root /store/audio/\n"
        ),
    )
    parser.add_argument(
        "dataset",
        help="Dataset name — determines the output filename (manifests/{dataset}.csv).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        metavar="PATH_OR_NAME",
        help=(
            "Path to the source manifest file.  Accepts .csv, .tsv, .xlsx, "
            ".xls, .parquet, .json, .jsonl.  When omitted, the dataset name "
            "is used for auto-detection in the manifests/ directory."
        ),
    )
    parser.add_argument(
        "--path-col",
        default="path",
        metavar="COLUMN",
        help=(
            "Column in the source manifest containing audio file paths "
            "(default: 'path')."
        ),
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        metavar="DIR",
        help=(
            "Root directory for relative audio paths.  Each relative path "
            "is resolved as <audio-root>/<path>."
        ),
    )
    args = parser.parse_args()

    normalize_manifest(
        dataset=args.dataset,
        manifest=args.manifest,
        path_col=args.path_col,
        audio_root=args.audio_root,
    )


if __name__ == "__main__":
    main()
