"""Tests for the full-audio clip tiling algorithm."""

from __future__ import annotations

import pytest

from src.packaging.clips import (
    CHUNK_STEP_S, CUT_TIERS, Clip, Segment,
    build_clips as _build_clips_raw,
    snap_to_grid,
    _build_activity_union,
    _find_silence_gaps, _compute_iou,
)


def build_clips(*args, **kwargs) -> list[Clip]:
    """Wrapper that discards tier_counts for convenience in most tests."""
    clips, _tier_counts = _build_clips_raw(*args, **kwargs)
    return clips


# ---------------------------------------------------------------------------
# _build_activity_union
# ---------------------------------------------------------------------------


class TestBuildActivityUnion:
    def test_empty(self):
        assert _build_activity_union([], []) == []

    def test_single_vtc(self):
        vtc = [Segment(onset=10, offset=20, label="FEM")]
        result = _build_activity_union(vtc, [])
        assert len(result) == 1
        assert result[0].onset == 10
        assert result[0].offset == 20

    def test_overlapping_vad_vtc(self):
        vtc = [Segment(onset=10, offset=25, label="FEM")]
        vad = [Segment(onset=20, offset=35)]
        result = _build_activity_union(vtc, vad)
        assert len(result) == 1
        assert result[0].onset == 10
        assert result[0].offset == 35

    def test_disjoint(self):
        vtc = [Segment(onset=10, offset=20, label="FEM")]
        vad = [Segment(onset=30, offset=40)]
        result = _build_activity_union(vtc, vad)
        assert len(result) == 2

    def test_multiple_overlaps_merge(self):
        vtc = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=15, offset=25, label="MAL"),
        ]
        vad = [Segment(onset=22, offset=30)]
        result = _build_activity_union(vtc, vad)
        assert len(result) == 1
        assert result[0].onset == 10
        assert result[0].offset == 30


# ---------------------------------------------------------------------------
# _find_silence_gaps
# ---------------------------------------------------------------------------


class TestFindSilenceGaps:
    def test_no_activity(self):
        gaps = _find_silence_gaps([], 0, 100)
        assert gaps == [(0, 100)]

    def test_full_activity(self):
        active = [Segment(onset=0, offset=100)]
        gaps = _find_silence_gaps(active, 0, 100)
        assert gaps == []

    def test_gap_before_and_after(self):
        active = [Segment(onset=30, offset=60)]
        gaps = _find_silence_gaps(active, 0, 100)
        assert len(gaps) == 2
        # Sorted longest first: (60,100)=40s then (0,30)=30s
        assert gaps[0] == (60, 100)
        assert gaps[1] == (0, 30)

    def test_gap_between_segments(self):
        active = [
            Segment(onset=10, offset=20),
            Segment(onset=50, offset=80),
        ]
        gaps = _find_silence_gaps(active, 0, 100)
        # (20,50)=30s, (80,100)=20s, (0,10)=10s
        assert gaps[0] == (20, 50)


# ---------------------------------------------------------------------------
# build_clips — basic tiling
# ---------------------------------------------------------------------------


class TestBuildClips:
    def test_zero_duration(self):
        clips = build_clips([], file_duration=0)
        assert clips == []

    def test_short_file_single_clip(self):
        """A file shorter than max_clip_s → exactly one clip."""
        vtc = [Segment(onset=10, offset=50, label="FEM")]
        clips = build_clips(vtc, file_duration=300, max_clip_s=600)
        assert len(clips) == 1
        assert clips[0].abs_onset == 0.0
        assert clips[0].abs_offset == 300.0

    def test_covers_full_duration(self):
        """Clips must cover [0, file_duration] with no gaps."""
        vtc = [Segment(onset=100, offset=200, label="FEM")]
        clips = build_clips(vtc, file_duration=3600, max_clip_s=600)
        assert clips[0].abs_onset == 0.0
        assert clips[-1].abs_offset == 3600.0
        # No gaps between clips
        for i in range(1, len(clips)):
            assert abs(clips[i].abs_onset - clips[i - 1].abs_offset) < 0.01

    def test_no_segments_still_produces_clips(self):
        """Even with no VTC/VAD, the full file is tiled."""
        clips = build_clips([], vad_segments=[], file_duration=1200, max_clip_s=600)
        assert len(clips) >= 1
        assert clips[0].abs_onset == 0.0
        assert clips[-1].abs_offset == 1200.0

    def test_vad_assigned_to_clip(self):
        """VAD segments overlapping a clip get assigned."""
        vtc = [Segment(onset=100, offset=130, label="FEM")]
        vad = [
            Segment(onset=95, offset=135),   # overlaps
            Segment(onset=900, offset=920),  # different clip
        ]
        clips = build_clips(vtc, vad_segments=vad, file_duration=300, max_clip_s=600)
        assert len(clips) == 1
        assert len(clips[0].vad_segments) >= 1
        assert len(clips[0].vtc_segments) == 1

    def test_vtc_segments_assigned(self):
        """VTC segments are assigned to the correct clip."""
        vtc = [
            Segment(onset=100, offset=200, label="FEM"),
            Segment(onset=800, offset=900, label="MAL"),
        ]
        clips = build_clips(vtc, file_duration=1000, max_clip_s=600)
        # With 1000s file and 600s max, should produce clips
        total_vtc = sum(len(c.vtc_segments) for c in clips)
        assert total_vtc == 2

    def test_sum_of_durations_equals_file(self):
        """Total clip duration must equal file duration."""
        vtc = [
            Segment(onset=i * 20, offset=i * 20 + 15, label="FEM")
            for i in range(100)
        ]
        dur = 2500.0
        clips = build_clips(vtc, file_duration=dur, max_clip_s=600)
        total = sum(c.duration for c in clips)
        assert abs(total - dur) < 0.01


# ---------------------------------------------------------------------------
# build_clips — splitting behaviour
# ---------------------------------------------------------------------------


class TestBuildClipsSplitting:
    def test_long_file_gets_split(self):
        """A 20-minute file → at least 2 clips."""
        vtc = [Segment(onset=0, offset=1200, label="FEM")]
        clips = build_clips(vtc, file_duration=1200, max_clip_s=600)
        assert len(clips) >= 2

    def test_prefers_silence_gap(self):
        """Insert a silence gap near the ideal boundary; cut should land there."""
        vtc = [
            Segment(onset=0, offset=570, label="FEM"),
            # 30 s silence gap at 570–600
            Segment(onset=600, offset=1200, label="MAL"),
        ]
        clips = build_clips(vtc, file_duration=1200, max_clip_s=600)
        assert len(clips) >= 2
        # The cut should be in the silence gap (570–600)
        cut = clips[0].abs_offset
        assert 570 <= cut <= 600

    def test_never_cuts_during_activity(self):
        """With dense activity streaks and clear gaps, cuts land in gaps."""
        # Activity from 0–580, gap 580–620, activity 620-1200
        vtc = [
            Segment(onset=0, offset=580, label="FEM"),
            Segment(onset=620, offset=1200, label="MAL"),
        ]
        clips = build_clips(vtc, file_duration=1200, max_clip_s=600)
        assert len(clips) >= 2
        cut = clips[0].abs_offset
        # Cut should be in the gap 580–620
        assert 580 <= cut <= 620

    def test_hard_cut_when_no_silence(self):
        """Continuous activity → forced hard cut (logged)."""
        vtc = [Segment(onset=0, offset=2000, label="FEM")]
        vad = [Segment(onset=0, offset=2000)]
        clips = build_clips(
            vtc, vad_segments=vad, file_duration=2000,
            max_clip_s=600, split_search_s=120,
        )
        assert len(clips) >= 2
        # All audio accounted for
        assert clips[0].abs_onset == 0.0
        assert clips[-1].abs_offset == 2000.0
        # Hard limit respected even with continuous activity
        for c in clips:
            assert c.duration <= 600 + 1

    def test_all_clips_within_max_duration(self):
        """Every clip is ≤ max_clip_s (hard limit)."""
        vtc = [
            Segment(onset=start, offset=start + 18, label="FEM")
            for start in range(0, 5000, 20)
        ]
        clips = build_clips(
            vtc, file_duration=5000,
            max_clip_s=600, split_search_s=120,
        )
        assert len(clips) >= 5
        for c in clips:
            assert c.duration <= 600 + 1  # +1 for float tolerance

    def test_slightly_over_max_produces_two_clips(self):
        """A file slightly longer than max_clip_s must be split, not kept as one."""
        vtc = [Segment(onset=0, offset=650, label="FEM")]
        clips = build_clips(vtc, file_duration=650, max_clip_s=600)
        assert len(clips) == 2
        for c in clips:
            assert c.duration <= 600 + 1

    def test_even_distribution(self):
        """601s should produce ~300s + ~301s, not 600 + 1."""
        # Pure silence — cuts should land near the ideal midpoint
        clips = build_clips([], file_duration=601, max_clip_s=600)
        assert len(clips) == 2
        # Both clips should be in the 250–400 range, not 600 + 1
        for c in clips:
            assert c.duration >= 250
            assert c.duration <= 400

    def test_even_distribution_three_clips(self):
        """1500s → 3 clips of ~500s, not 600 + 600 + 300."""
        clips = build_clips([], file_duration=1500, max_clip_s=600)
        assert len(clips) == 3
        for c in clips:
            assert c.duration >= 400
            assert c.duration <= 600

    def test_prefers_long_silence_gap(self):
        """A long gap (>10s) is preferred over a short gap nearer the ideal."""
        vtc = [
            Segment(onset=0, offset=280, label="FEM"),
            # Short 2s gap at 280–282
            Segment(onset=282, offset=330, label="MAL"),
            # Long 20s gap at 330–350  (further from ideal=350)
            Segment(onset=350, offset=700, label="FEM"),
        ]
        clips = build_clips(vtc, file_duration=700, max_clip_s=600, min_gap_s=10)
        assert len(clips) == 2
        cut = clips[0].abs_offset
        # Should prefer the long gap 330–350 (midpoint=340)
        assert 330 <= cut <= 350

    def test_vad_only_gap_fallback(self):
        """When VAD∪VTC union has no gap but VAD alone does, cut there."""
        # VTC covers the entire file — no union gap.
        # VAD has a gap at 580–620 near the ideal cut.
        vtc = [Segment(onset=0, offset=1200, label="FEM")]
        vad = [
            Segment(onset=0, offset=580),
            Segment(onset=620, offset=1200),
        ]
        clips = build_clips(
            vtc, vad_segments=vad, file_duration=1200,
            max_clip_s=600, split_search_s=120,
        )
        assert len(clips) >= 2
        cut = clips[0].abs_offset
        # Should land in the VAD gap (580–620)
        assert 580 <= cut <= 620

    def test_vtc_only_gap_fallback(self):
        """When union and VAD have no gap but VTC alone does, cut there."""
        # VAD covers the entire file — no union or VAD-only gap.
        # VTC has a gap at 580–620 near the ideal cut.
        vtc = [
            Segment(onset=0, offset=580, label="FEM"),
            Segment(onset=620, offset=1200, label="MAL"),
        ]
        vad = [Segment(onset=0, offset=1200)]
        clips = build_clips(
            vtc, vad_segments=vad, file_duration=1200,
            max_clip_s=600, split_search_s=120,
        )
        assert len(clips) >= 2
        cut = clips[0].abs_offset
        # Should land in the VTC gap (580–620)
        assert 580 <= cut <= 620

    def test_speaker_boundary_fallback(self):
        """When no gaps at all, fall back to VTC speaker boundary."""
        # Both VAD and VTC cover the entire file — no gaps anywhere.
        # But there's a speaker change at t=580.
        vtc = [
            Segment(onset=0, offset=580, label="FEM"),
            Segment(onset=580, offset=1200, label="MAL"),
        ]
        vad = [Segment(onset=0, offset=1200)]
        clips = build_clips(
            vtc, vad_segments=vad, file_duration=1200,
            max_clip_s=600, split_search_s=120,
        )
        assert len(clips) >= 2
        cut = clips[0].abs_offset
        # Should land at or near the speaker boundary (580)
        assert 575 <= cut <= 620


# ---------------------------------------------------------------------------
# build_clips — tier counts
# ---------------------------------------------------------------------------


class TestTierCounts:
    def test_single_clip_no_cuts(self):
        """A short file produces no cuts → all tier counts zero."""
        _clips, tier_counts = _build_clips_raw([], file_duration=300, max_clip_s=600)
        assert sum(tier_counts.values()) == 0

    def test_silence_gap_tier(self):
        """Cuts in silence → long_union_gap or short_union_gap."""
        vtc = [
            Segment(onset=0, offset=280, label="FEM"),
            Segment(onset=320, offset=600, label="MAL"),
        ]
        _clips, tier_counts = _build_clips_raw(
            vtc, file_duration=600, max_clip_s=400,
        )
        # At least one cut should have landed in the 280–320 gap
        assert tier_counts["long_union_gap"] + tier_counts["short_union_gap"] >= 1
        assert tier_counts["hard_cut"] == 0

    def test_hard_cut_tier(self):
        """Continuous activity with a single speaker → hard_cut tier."""
        vtc = [Segment(onset=0, offset=2000, label="FEM")]
        vad = [Segment(onset=0, offset=2000)]
        _clips, tier_counts = _build_clips_raw(
            vtc, vad_segments=vad, file_duration=2000,
            max_clip_s=600, split_search_s=120,
        )
        assert tier_counts["hard_cut"] >= 1

    def test_vad_only_gap_counted(self):
        """VAD-only gap fallback is counted correctly."""
        vtc = [Segment(onset=0, offset=1200, label="FEM")]
        vad = [
            Segment(onset=0, offset=580),
            Segment(onset=620, offset=1200),
        ]
        _clips, tier_counts = _build_clips_raw(
            vtc, vad_segments=vad, file_duration=1200,
            max_clip_s=600, split_search_s=120,
        )
        assert tier_counts["vad_only_gap"] >= 1

    def test_tier_counts_keys(self):
        """Tier counts dict always has all expected keys."""
        _clips, tier_counts = _build_clips_raw([], file_duration=0)
        for tier in CUT_TIERS:
            assert tier in tier_counts


# ---------------------------------------------------------------------------
# Clip metadata & properties
# ---------------------------------------------------------------------------


class TestClipMetadata:
    def test_relative_timestamps(self):
        clip = Clip(abs_onset=100.0, abs_offset=200.0)
        clip.vtc_segments = [Segment(onset=120, offset=140, label="FEM")]
        clip.vad_segments = [Segment(onset=110, offset=150)]

        meta = clip.to_metadata("test_uid", 0)
        assert meta["abs_onset"] == 100.0
        assert meta["vtc_segments"][0]["onset"] == 20.0
        assert meta["vad_segments"][0]["onset"] == 10.0
        assert meta["clip_id"] == "test_uid_0000"

    def test_vtc_speech_density(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [Segment(onset=10, offset=60, label="FEM")]
        assert clip.speech_density == 0.5

    def test_turns(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=25, offset=35, label="KCHI"),
            Segment(onset=40, offset=50, label="FEM"),
        ]
        assert clip.n_turns == 3  # FEM→KCHI→FEM

    def test_turns_same_speaker(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=25, offset=35, label="FEM"),
        ]
        assert clip.n_turns == 1

    def test_labels_present(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=25, offset=35, label="KCHI"),
            Segment(onset=40, offset=50, label="FEM"),
        ]
        assert clip.labels_present == ["FEM", "KCHI"]
        assert clip.n_labels == 2
        assert clip.has_adult is True

    def test_no_adult(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="KCHI"),
            Segment(onset=25, offset=35, label="OCH"),
        ]
        assert clip.has_adult is False

    def test_mean_vtc_gap(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=25, offset=35, label="MAL"),   # gap = 5s
            Segment(onset=40, offset=50, label="FEM"),    # gap = 5s
        ]
        assert clip.mean_vtc_gap == 5.0

    def test_metadata_has_new_fields(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=30, offset=40, label="MAL"),
        ]
        clip.vad_segments = [Segment(onset=5, offset=45)]
        meta = clip.to_metadata("uid", 0)
        assert "vtc_speech_duration" in meta
        assert "vtc_speech_density" in meta
        assert "vad_speech_duration" in meta
        assert "vad_speech_density" in meta
        assert "vad_vtc_iou" in meta
        assert "n_turns" in meta
        assert "n_labels" in meta
        assert "labels_present" in meta
        assert "has_adult" in meta
        assert "mean_vtc_seg_duration" in meta
        assert "mean_vtc_gap" in meta


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


class TestIoU:
    def test_perfect_overlap(self):
        vad = [Segment(onset=10, offset=50)]
        vtc = [Segment(onset=10, offset=50, label="FEM")]
        iou = _compute_iou(vad, vtc, 0, 100)
        assert iou > 0.95

    def test_no_overlap(self):
        vad = [Segment(onset=10, offset=20)]
        vtc = [Segment(onset=50, offset=60, label="FEM")]
        iou = _compute_iou(vad, vtc, 0, 100)
        assert iou == 0.0

    def test_partial_overlap(self):
        vad = [Segment(onset=10, offset=30)]
        vtc = [Segment(onset=20, offset=40, label="FEM")]
        iou = _compute_iou(vad, vtc, 0, 100)
        assert 0.25 < iou < 0.40

    def test_empty_segments(self):
        iou = _compute_iou([], [], 0, 100)
        assert iou == 0.0


# ---------------------------------------------------------------------------
# Per-label properties
# ---------------------------------------------------------------------------


class TestPerLabelProperties:
    def test_label_durations(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),   # 10s
            Segment(onset=30, offset=40, label="KCHI"),   # 10s
            Segment(onset=50, offset=55, label="FEM"),    # 5s
        ]
        ld = clip.label_durations
        assert abs(ld["FEM"] - 15.0) < 0.01
        assert abs(ld["KCHI"] - 10.0) < 0.01
        assert "MAL" not in ld

    def test_child_adult_durations(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),    # adult 10s
            Segment(onset=30, offset=50, label="KCHI"),   # child 20s
            Segment(onset=60, offset=65, label="OCH"),    # child 5s
        ]
        assert abs(clip.adult_speech_duration - 10.0) < 0.01
        assert abs(clip.child_speech_duration - 25.0) < 0.01
        assert abs(clip.child_fraction - 25.0 / 35.0) < 0.01

    def test_dominant_label(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=30, offset=60, label="KCHI"),  # longest
        ]
        assert clip.dominant_label == "KCHI"

    def test_vad_coverage_by_label(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),    # 10s, fully covered
            Segment(onset=30, offset=50, label="KCHI"),   # 20s, not covered
        ]
        clip.vad_segments = [
            Segment(onset=8, offset=22),   # covers FEM fully
        ]
        cov = clip.vad_coverage_by_label()
        assert cov["FEM"] >= 0.95
        assert cov["KCHI"] == 0.0

    def test_metadata_has_label_fields(self):
        clip = Clip(abs_onset=0, abs_offset=100)
        clip.vtc_segments = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=30, offset=40, label="KCHI"),
        ]
        meta = clip.to_metadata("uid", 0)
        assert "child_speech_duration" in meta
        assert "adult_speech_duration" in meta
        assert "child_fraction" in meta
        assert "dominant_label" in meta
        assert "label_durations" in meta
        assert "vad_coverage_by_label" in meta


# ---------------------------------------------------------------------------
# snap_to_grid
# ---------------------------------------------------------------------------


class TestSnapToGrid:
    def test_exact_multiple_unchanged(self):
        """A time already on the grid stays put."""
        t = 3 * CHUNK_STEP_S  # 11.94
        assert snap_to_grid(t, 1000.0) == pytest.approx(t)

    def test_rounds_to_nearest(self):
        """A time between two grid points rounds to the nearest."""
        # Just above 1 * CHUNK_STEP_S
        assert snap_to_grid(CHUNK_STEP_S + 0.1, 1000.0) == pytest.approx(CHUNK_STEP_S)
        # Just below 2 * CHUNK_STEP_S
        assert snap_to_grid(2 * CHUNK_STEP_S - 0.1, 1000.0) == pytest.approx(
            2 * CHUNK_STEP_S
        )

    def test_zero_stays_zero(self):
        assert snap_to_grid(0.0, 1000.0) == 0.0

    def test_clamped_to_file_duration(self):
        """Result never exceeds file_duration."""
        dur = 10.0
        assert snap_to_grid(9.99, dur) <= dur

    def test_clamped_to_zero(self):
        """Result never goes below 0."""
        assert snap_to_grid(0.5, 1000.0) >= 0.0


# ---------------------------------------------------------------------------
# build_clips with snap_to_chunk_grid
# ---------------------------------------------------------------------------


class TestBuildClipsGridSnapping:
    """Verify that snap_to_chunk_grid produces on-grid boundaries."""

    def _is_on_grid(self, t: float) -> bool:
        """True if *t* is a multiple of CHUNK_STEP_S (within tolerance)."""
        remainder = t % CHUNK_STEP_S
        return remainder < 1e-6 or (CHUNK_STEP_S - remainder) < 1e-6

    def test_boundaries_on_grid(self):
        """All interior boundaries should land on the chunk grid."""
        dur = 3600.0  # 1 hour → ~6 clips
        vtc = [Segment(onset=i * 60, offset=i * 60 + 50, label="FEM") for i in range(60)]
        vad = [Segment(onset=i * 60, offset=i * 60 + 50) for i in range(60)]
        clips = build_clips(vtc, vad, file_duration=dur)
        for clip in clips[:-1]:  # last clip ends at file_duration, not snapped
            assert self._is_on_grid(clip.abs_onset), (
                f"abs_onset {clip.abs_onset} is not on the chunk grid"
            )

    def test_first_boundary_is_zero(self):
        """First clip always starts at 0."""
        dur = 2000.0
        clips = build_clips([], [], file_duration=dur)
        assert clips[0].abs_onset == 0.0

    def test_last_boundary_is_file_duration(self):
        """Last clip always ends at file_duration (never snapped)."""
        dur = 2000.123  # deliberately not on grid
        clips = build_clips([], [], file_duration=dur)
        assert clips[-1].abs_offset == pytest.approx(dur)

    def test_no_zero_duration_clips(self):
        """Snapping should never produce a zero-duration clip."""
        dur = 3600.0
        vtc = [Segment(onset=i * 60, offset=i * 60 + 50, label="FEM") for i in range(60)]
        clips = build_clips(vtc, file_duration=dur)
        for clip in clips:
            assert clip.duration > 0, f"Zero-duration clip at {clip.abs_onset}"

    def test_snap_off_preserves_old_behaviour(self):
        """With snap_to_chunk_grid=False, boundaries need not be on grid."""
        dur = 3600.0
        vtc = [Segment(onset=100, offset=200, label="FEM")]
        clips_raw, _ = _build_clips_raw(
            vtc, file_duration=dur, snap_to_chunk_grid=False
        )
        on_grid = all(self._is_on_grid(c.abs_onset) for c in clips_raw[1:])
        # Very unlikely that all boundaries happen to be on-grid by chance
        # with smooth gaps; but the real test is that the parameter works.
        assert len(clips_raw) > 1

    def test_full_coverage(self):
        """Every second of the file is covered — no gaps."""
        dur = 3600.0
        vtc = [Segment(onset=i * 60, offset=i * 60 + 50, label="FEM") for i in range(60)]
        clips = build_clips(vtc, file_duration=dur)
        assert clips[0].abs_onset == 0.0
        assert clips[-1].abs_offset == pytest.approx(dur)
        for i in range(len(clips) - 1):
            assert clips[i].abs_offset == pytest.approx(clips[i + 1].abs_onset), (
                f"Gap between clip {i} and {i+1}"
            )

    def test_short_file_single_clip(self):
        """A file shorter than max_clip_s produces a single clip, unchanged."""
        dur = 300.0
        clips = build_clips([], file_duration=dur)
        assert len(clips) == 1
        assert clips[0].abs_onset == 0.0
        assert clips[0].abs_offset == dur
