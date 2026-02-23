"""Tests for src.core.parallel — parallel VAD driver."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import requires_tenvad

from src.core.parallel import run_vad_parallel


@requires_tenvad
class TestRunVadParallel:
    def test_single_file(self, good_book_wavs: list[Path]):
        """Run VAD on one file with 1 worker."""
        meta, segs = run_vad_parallel(
            good_book_wavs[:1], hop_size=256, threshold=0.5, workers=1,
        )
        assert len(meta) == 1
        assert meta[0]["success"] is True
        assert len(segs) > 0

    def test_multiple_files_parallel(self, good_book_wavs: list[Path]):
        """Run VAD on multiple files with 2 workers."""
        wavs = good_book_wavs[:3]
        meta, segs = run_vad_parallel(
            wavs, hop_size=256, threshold=0.5, workers=2,
        )
        assert len(meta) == len(wavs)
        assert all(m["success"] for m in meta)

    def test_with_checkpointing(self, good_book_wavs: list[Path], tmp_path: Path):
        """Verify checkpointing dir doesn't crash (interval too high to actually trigger)."""
        meta, segs = run_vad_parallel(
            good_book_wavs[:1], hop_size=256, threshold=0.5, workers=1,
            checkpoint_dir=tmp_path, checkpoint_interval=1,
        )
        assert len(meta) == 1
