"""Tests for src.core.checkpoint — checkpoint save / clear."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.core.checkpoint import (
    CKPT_META,
    CKPT_SEGS,
    clear_checkpoint,
    save_checkpoint,
)


class TestSaveCheckpoint:
    def test_saves_meta(self, tmp_path: Path):
        rows = [{"file_id": "a", "success": True}]
        save_checkpoint(tmp_path, rows, [])
        assert (tmp_path / CKPT_META).exists()
        df = pl.read_parquet(tmp_path / CKPT_META)
        assert len(df) == 1
        assert df["file_id"][0] == "a"

    def test_saves_segs(self, tmp_path: Path):
        segs = [{"file_id": "a", "onset": 0.0, "offset": 1.0}]
        save_checkpoint(tmp_path, [], segs)
        assert (tmp_path / CKPT_SEGS).exists()
        df = pl.read_parquet(tmp_path / CKPT_SEGS)
        assert len(df) == 1

    def test_empty_rows_no_file(self, tmp_path: Path):
        save_checkpoint(tmp_path, [], [])
        assert not (tmp_path / CKPT_META).exists()
        assert not (tmp_path / CKPT_SEGS).exists()


class TestClearCheckpoint:
    def test_removes_files(self, tmp_path: Path):
        (tmp_path / CKPT_META).touch()
        (tmp_path / CKPT_SEGS).touch()
        clear_checkpoint(tmp_path)
        assert not (tmp_path / CKPT_META).exists()
        assert not (tmp_path / CKPT_SEGS).exists()

    def test_no_error_if_missing(self, tmp_path: Path):
        # Should not raise even if files don't exist
        clear_checkpoint(tmp_path)
