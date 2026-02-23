"""Tests for src.core.regions — activity-region construction."""

from __future__ import annotations

import pytest

from tests.conftest import _TORCHCODEC_OK, requires_torchcodec

if not _TORCHCODEC_OK:
    pytestmark = requires_torchcodec
    # Provide stubs so the module can be collected without import errors.
    merge_into_activity_regions = None  # type: ignore[assignment]
    activity_region_coverage = None  # type: ignore[assignment]
else:
    from src.core.regions import (
        activity_region_coverage,
        merge_into_activity_regions,
    )


# ---------------------------------------------------------------------------
# merge_into_activity_regions
# ---------------------------------------------------------------------------


class TestMergeIntoActivityRegions:
    def test_empty_vad(self):
        assert merge_into_activity_regions([], 100.0) == []

    def test_single_segment(self):
        regions = merge_into_activity_regions([(10.0, 20.0)], 100.0)
        assert len(regions) == 1
        onset, offset = regions[0]
        # Should be padded: 10-5=5, 20+5=25
        assert onset == pytest.approx(5.0)
        assert offset == pytest.approx(25.0)

    def test_padding_clipped_to_zero(self):
        regions = merge_into_activity_regions([(2.0, 5.0)], 100.0, pad_s=5.0)
        assert regions[0][0] == 0.0

    def test_padding_clipped_to_file_end(self):
        regions = merge_into_activity_regions([(95.0, 98.0)], 100.0, pad_s=5.0)
        assert regions[0][1] == 100.0

    def test_close_segments_merged(self):
        """Segments within merge_gap_s should become one region."""
        vad = [(10.0, 15.0), (20.0, 25.0)]
        regions = merge_into_activity_regions(vad, 200.0, merge_gap_s=10.0, pad_s=0.0)
        assert len(regions) == 1
        assert regions[0] == pytest.approx((10.0, 25.0), abs=0.01)

    def test_distant_segments_separate(self):
        """Segments far apart should remain separate regions."""
        vad = [(10.0, 15.0), (100.0, 110.0)]
        regions = merge_into_activity_regions(vad, 200.0, merge_gap_s=30.0, pad_s=5.0)
        assert len(regions) == 2

    def test_many_small_segments(self):
        """Many small segments close together → one large region."""
        vad = [(i * 2.0, i * 2.0 + 0.5) for i in range(20)]
        regions = merge_into_activity_regions(vad, 100.0, merge_gap_s=5.0, pad_s=2.0)
        # All 20 segments are within gap → should merge into 1 region
        assert len(regions) == 1

    def test_full_file_coverage(self):
        """Many dense segments → coverage close to 100%."""
        vad = [(i * 1.0, i * 1.0 + 0.9) for i in range(100)]
        regions = merge_into_activity_regions(vad, 100.0, merge_gap_s=5.0, pad_s=5.0)
        cov = activity_region_coverage(regions, 100.0)
        assert cov == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# activity_region_coverage
# ---------------------------------------------------------------------------


class TestActivityRegionCoverage:
    def test_empty(self):
        assert activity_region_coverage([], 100.0) == 0.0

    def test_full_file(self):
        assert activity_region_coverage([(0.0, 100.0)], 100.0) == pytest.approx(1.0)

    def test_half_file(self):
        assert activity_region_coverage([(0.0, 50.0)], 100.0) == pytest.approx(0.5)

    def test_zero_duration_file(self):
        assert activity_region_coverage([(0.0, 1.0)], 0.0) == 1.0

    def test_multiple_regions(self):
        regions = [(0.0, 10.0), (50.0, 60.0)]
        assert activity_region_coverage(regions, 100.0) == pytest.approx(0.2)
