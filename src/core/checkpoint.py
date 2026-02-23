"""Checkpoint helpers for resumable long-running jobs."""

from __future__ import annotations

from pathlib import Path

import polars as pl

CKPT_META = "_checkpoint_meta.parquet"
CKPT_SEGS = "_checkpoint_segs.parquet"


def save_checkpoint(
    directory: Path,
    meta_rows: list[dict],
    seg_rows: list[dict],
) -> None:
    """Write in-progress results so the run can be resumed."""
    if meta_rows:
        pl.DataFrame(meta_rows).write_parquet(
            directory / CKPT_META, compression="zstd"
        )
    if seg_rows:
        pl.DataFrame(seg_rows).write_parquet(
            directory / CKPT_SEGS, compression="zstd"
        )


def clear_checkpoint(directory: Path) -> None:
    """Remove checkpoint files after a successful run."""
    for name in (CKPT_META, CKPT_SEGS):
        p = directory / name
        if p.exists():
            p.unlink()
