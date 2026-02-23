"""Tests for src.core.vad_processing — VAD helpers and single-file processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_tenvad

from src.core.vad_processing import (
    get_runs,
    resample_block,
    runs_to_segments,
    segment_stats,
    vad_error_metadata,
    process_vad_file,
)


# ---------------------------------------------------------------------------
# get_runs
# ---------------------------------------------------------------------------


class TestGetRuns:
    def test_empty(self):
        sp, si = get_runs(np.array([], dtype=np.uint8))
        assert sp.shape == (0, 2)
        assert si.shape == (0, 2)

    def test_all_speech(self):
        flags = np.ones(100, dtype=np.uint8)
        sp, si = get_runs(flags)
        assert len(sp) == 1
        assert sp[0, 0] == 0
        assert sp[0, 1] == 100
        assert len(si) == 0

    def test_all_silence(self):
        flags = np.zeros(100, dtype=np.uint8)
        sp, si = get_runs(flags)
        assert len(sp) == 0
        assert len(si) == 1
        assert si[0, 0] == 0
        assert si[0, 1] == 100

    def test_alternating(self):
        # 10 speech, 10 silence, 10 speech
        flags = np.array([1]*10 + [0]*10 + [1]*10, dtype=np.uint8)
        sp, si = get_runs(flags)
        assert len(sp) == 2
        assert len(si) == 1

    def test_single_frame(self):
        sp, si = get_runs(np.array([1], dtype=np.uint8))
        assert len(sp) == 1
        assert len(si) == 0

    def test_speech_silence_boundary(self):
        """Verify run boundaries are correct at transitions."""
        flags = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
        sp, si = get_runs(flags)
        assert list(sp[0]) == [0, 2]
        assert list(sp[1]) == [4, 5]
        assert list(si[0]) == [2, 4]


# ---------------------------------------------------------------------------
# runs_to_segments
# ---------------------------------------------------------------------------


class TestRunsToSegments:
    def test_empty(self):
        assert runs_to_segments(np.empty((0, 2), dtype=int), 256, 16000) == []

    def test_single_run(self):
        runs = np.array([[0, 100]])
        segs = runs_to_segments(runs, 256, 16000)
        assert len(segs) == 1
        assert set(segs[0].keys()) == {"onset", "offset", "duration"}
        assert segs[0]["onset"] == 0.0
        expected_offset = round(100 * 256 / 16000, 3)
        assert segs[0]["offset"] == expected_offset
        assert segs[0]["duration"] == expected_offset

    def test_multiple_runs(self):
        runs = np.array([[0, 10], [20, 30]])
        segs = runs_to_segments(runs, 256, 16000)
        assert len(segs) == 2
        assert segs[0]["offset"] <= segs[1]["onset"]


# ---------------------------------------------------------------------------
# segment_stats
# ---------------------------------------------------------------------------


class TestSegmentStats:
    def test_empty(self):
        stats = segment_stats([])
        assert stats["num"] == 0
        assert stats["sum"] == 0.0

    def test_single(self):
        stats = segment_stats([5.0])
        assert stats["num"] == 1
        assert stats["max"] == 5.0
        assert stats["min"] == 5.0
        assert stats["avg"] == 5.0
        assert stats["sum"] == 5.0

    def test_multiple(self):
        stats = segment_stats([1.0, 2.0, 3.0])
        assert stats["num"] == 3
        assert stats["max"] == 3.0
        assert stats["min"] == 1.0
        assert stats["avg"] == pytest.approx(2.0)
        assert stats["sum"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# resample_block
# ---------------------------------------------------------------------------


class TestResampleBlock:
    def test_same_rate(self):
        data = np.arange(100, dtype=np.int16)
        result = resample_block(data, 16000, 16000)
        np.testing.assert_array_equal(result, data)

    def test_downsample(self):
        data = np.arange(1000, dtype=np.int16)
        result = resample_block(data, 32000, 16000)
        assert len(result) == 500

    def test_upsample(self):
        data = np.arange(100, dtype=np.int16)
        result = resample_block(data, 8000, 16000)
        assert len(result) == 200

    def test_output_dtype(self):
        data = np.arange(100, dtype=np.int16)
        result = resample_block(data, 8000, 16000)
        assert result.dtype == np.int16


# ---------------------------------------------------------------------------
# vad_error_metadata
# ---------------------------------------------------------------------------


class TestVadErrorMetadata:
    def test_structure(self):
        row = vad_error_metadata("/some/path.wav", "boom")
        assert row["success"] is False
        assert row["error"] == "boom"
        assert row["file_id"] == "path"
        assert row["duration"] == 0.0

    def test_all_keys_present(self):
        row = vad_error_metadata("/a.wav", "err")
        expected_keys = {
            "success", "path", "file_id", "duration", "original_sr",
            "speech_ratio", "n_speech_segments", "n_silence_segments",
            "speech_max", "speech_min", "speech_sum", "speech_num", "speech_avg",
            "nospch_max", "nospch_min", "nospch_sum", "nospch_num", "nospch_avg",
            "has_long_segment", "error",
        }
        assert set(row.keys()) == expected_keys


# ---------------------------------------------------------------------------
# process_vad_file  (integration: requires real audio)
# ---------------------------------------------------------------------------


@requires_tenvad
class TestProcessFile:
    def test_on_good_book_file(self, good_book_wavs: list[Path]):
        """End-to-end test on a real short audio file."""
        wav = good_book_wavs[0]
        meta, segs = process_vad_file((wav, 256, 0.5))
        assert meta["success"] is True
        assert meta["duration"] > 0
        assert meta["file_id"] == wav.stem
        assert meta["speech_ratio"] >= 0.0
        # Audiobook files should have detectable speech
        assert meta["n_speech_segments"] > 0
        for seg in segs:
            assert set(seg.keys()) == {"file_id", "onset", "offset", "duration"}
            assert seg["duration"] > 0

    def test_on_short_fail_file(self, short_fail_wavs: list[Path]):
        """Short files should still process without error."""
        wav = short_fail_wavs[0]
        meta, segs = process_vad_file((wav, 256, 0.5))
        assert meta["success"] is True
        assert meta["duration"] > 0

    def test_nonexistent_file(self, tmp_path: Path):
        """Nonexistent file should return error metadata, not raise."""
        meta, segs = process_vad_file((tmp_path / "nope.wav", 256, 0.5))
        assert meta["success"] is False
        assert "error" in meta
        assert segs == []
