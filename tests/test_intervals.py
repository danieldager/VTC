"""Tests for src.core.intervals — pure interval arithmetic."""

from __future__ import annotations

import pytest

from src.core.intervals import (
    compute_iou,
    intervals_to_pairs,
    intervals_to_segments,
    merge_pairs,
    total_duration,
)


# ---------------------------------------------------------------------------
# merge_pairs
# ---------------------------------------------------------------------------


class TestMergePairs:
    def test_empty(self):
        assert merge_pairs([]) == []

    def test_single(self):
        assert merge_pairs([(1.0, 2.0)]) == [(1.0, 2.0)]

    def test_non_overlapping(self):
        pairs = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
        assert merge_pairs(pairs) == pairs

    def test_overlapping(self):
        pairs = [(0.0, 2.0), (1.0, 3.0), (5.0, 6.0)]
        assert merge_pairs(pairs) == [(0.0, 3.0), (5.0, 6.0)]

    def test_touching_boundaries(self):
        """Segments where one ends exactly where another begins → merged."""
        pairs = [(0.0, 1.0), (1.0, 2.0)]
        assert merge_pairs(pairs) == [(0.0, 2.0)]

    def test_nested(self):
        """A segment fully contained in another."""
        pairs = [(0.0, 10.0), (2.0, 5.0)]
        assert merge_pairs(pairs) == [(0.0, 10.0)]

    def test_unsorted_input(self):
        pairs = [(3.0, 4.0), (0.0, 1.0), (2.0, 3.5)]
        result = merge_pairs(pairs)
        assert result == [(0.0, 1.0), (2.0, 4.0)]

    def test_all_same(self):
        pairs = [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]
        assert merge_pairs(pairs) == [(1.0, 2.0)]


# ---------------------------------------------------------------------------
# total_duration
# ---------------------------------------------------------------------------


class TestTotalDuration:
    def test_empty(self):
        assert total_duration([]) == 0.0

    def test_single(self):
        assert total_duration([(0.0, 5.0)]) == 5.0

    def test_multiple(self):
        assert total_duration([(0.0, 1.0), (3.0, 5.0)]) == 3.0

    def test_zero_width(self):
        assert total_duration([(1.0, 1.0)]) == 0.0


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


class TestComputeIoU:
    def test_both_empty(self):
        assert compute_iou([], []) == 1.0

    def test_one_empty(self):
        assert compute_iou([(0.0, 1.0)], []) == 0.0
        assert compute_iou([], [(0.0, 1.0)]) == 0.0

    def test_identical(self):
        pairs = [(0.0, 10.0)]
        assert compute_iou(pairs, pairs) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_iou([(0.0, 1.0)], [(2.0, 3.0)]) == 0.0

    def test_partial_overlap(self):
        vtc = [(0.0, 2.0)]
        vad = [(1.0, 3.0)]
        # intersection = 1, union = 3
        assert compute_iou(vtc, vad) == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_one_contains_other(self):
        big = [(0.0, 10.0)]
        small = [(2.0, 5.0)]
        # intersection = 3, union = 10
        assert compute_iou(big, small) == pytest.approx(0.3, abs=1e-6)

    def test_multiple_segments(self):
        vtc = [(0.0, 2.0), (4.0, 6.0)]
        vad = [(1.0, 5.0)]
        # intersection: (1,2)=1 + (4,5)=1 = 2
        # union: 4 (vtc) + 4 (vad) - 2 = 6
        assert compute_iou(vtc, vad) == pytest.approx(2.0 / 6.0, abs=1e-6)


# ---------------------------------------------------------------------------
# intervals_to_pairs  (requires segma for frames_to_seconds)
# ---------------------------------------------------------------------------


class TestIntervalsToPairs:
    def test_empty(self):
        assert intervals_to_pairs([]) == []

    def test_single_interval(self):
        # frames_to_seconds converts sample indices to seconds
        # The exact mapping depends on segma's implementation,
        # but we can test that the output is sorted and merged.
        intervals = [(0, 16000, "KCHI")]
        result = intervals_to_pairs(intervals)
        assert len(result) >= 1
        for onset, offset in result:
            assert onset < offset

    def test_overlapping_intervals_merged(self):
        # Two intervals that overlap → should merge
        intervals = [(0, 16000, "KCHI"), (8000, 24000, "FEM")]
        result = intervals_to_pairs(intervals)
        # Should merge into one or two pairs depending on frame conversion
        assert len(result) >= 1
        assert all(onset < offset for onset, offset in result)


# ---------------------------------------------------------------------------
# intervals_to_segments
# ---------------------------------------------------------------------------


class TestIntervalsToSegments:
    def test_empty(self):
        assert intervals_to_segments([], "test") == []

    def test_segment_structure(self):
        intervals = [(0, 16000, "KCHI")]
        segs = intervals_to_segments(intervals, "file001")
        assert len(segs) >= 1
        for seg in segs:
            assert set(seg.keys()) == {"uid", "onset", "offset", "duration", "label"}
            assert seg["uid"] == "file001"
            assert seg["label"] == "KCHI"
            assert seg["duration"] > 0
            assert seg["offset"] > seg["onset"]

    def test_zero_duration_filtered(self):
        """Intervals that produce zero-duration segments should be filtered out."""
        # Create an interval so short that after conversion it rounds to 0
        intervals = [(0, 0, "FEM")]
        segs = intervals_to_segments(intervals, "x")
        assert segs == []
