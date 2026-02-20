"""Tests for scripts.core.metadata — VTC metadata row builders."""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest

from scripts.core.metadata import (
    _EMPTY_VTC_META,
    load_vad_reference,
    vtc_error_row,
    vtc_meta_row,
)


class TestVtcErrorRow:
    def test_structure(self):
        row = vtc_error_row("file001", "some error")
        assert row["uid"] == "file001"
        assert row["vtc_status"] == "error"
        assert row["error"] == "some error"

    def test_has_all_keys(self):
        row = vtc_error_row("x", "e")
        for key in _EMPTY_VTC_META:
            assert key in row


class TestVtcMetaRow:
    def test_basic(self):
        segments = [
            {"label": "KCHI", "duration": 1.0},
            {"label": "KCHI", "duration": 2.0},
            {"label": "FEM", "duration": 3.0},
        ]
        row = vtc_meta_row("uid1", 0.4, 0.85, "met_target", segments, 0.9, 0.6)
        assert row["uid"] == "uid1"
        assert row["vtc_threshold"] == 0.4
        assert row["vtc_vad_iou"] == 0.85
        assert row["vtc_status"] == "met_target"
        assert row["vtc_speech_dur"] == pytest.approx(6.0)
        assert row["vtc_n_segments"] == 3
        counts = json.loads(row["vtc_label_counts"])
        assert counts["KCHI"] == 2
        assert counts["FEM"] == 1

    def test_empty_segments(self):
        row = vtc_meta_row("uid2", 0.5, 1.0, "vad_empty", [], 0.0, 0.0)
        assert row["vtc_speech_dur"] == 0.0
        assert row["vtc_n_segments"] == 0
        assert json.loads(row["vtc_label_counts"]) == {}


class TestLoadVadReference:
    def test_empty_dir(self, tmp_path: Path):
        result = load_vad_reference(tmp_path)
        assert result == {}

    def test_loads_from_vad_merged(self, tmp_path: Path):
        vad_dir = tmp_path / "vad_merged"
        vad_dir.mkdir()
        df = pl.DataFrame({
            "uid": ["a", "a", "b"],
            "onset": [0.0, 5.0, 1.0],
            "offset": [3.0, 8.0, 4.0],
        })
        df.write_parquet(vad_dir / "segments.parquet")

        result = load_vad_reference(tmp_path)
        assert "a" in result
        assert "b" in result
        # "a" should have 2 pairs (possibly merged if close enough)
        assert len(result["a"]) >= 1
        assert len(result["b"]) == 1
        assert result["b"][0] == (1.0, 4.0)
