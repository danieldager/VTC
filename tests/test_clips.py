"""Tests for the VTC-based clip building algorithm."""

from __future__ import annotations

import pytest

from src.packaging.clips import Clip, Segment, build_clips, _merge_segments, _compute_iou


# ---------------------------------------------------------------------------
# _merge_segments
# ---------------------------------------------------------------------------


class TestMergeSegments:
    def test_empty(self):
        assert _merge_segments([], max_gap=10) == []

    def test_single(self):
        segs = [Segment(onset=10, offset=20, label="FEM")]
        regions = _merge_segments(segs, max_gap=5)
        assert len(regions) == 1
        assert regions[0][0] == 10
        assert regions[0][1] == 20

    def test_merge_within_gap(self):
        segs = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=25, offset=35, label="MAL"),
        ]
        regions = _merge_segments(segs, max_gap=10)
        assert len(regions) == 1
        assert regions[0][0] == 10
        assert regions[0][1] == 35

    def test_split_beyond_gap(self):
        segs = [
            Segment(onset=10, offset=20, label="FEM"),
            Segment(onset=50, offset=60, label="MAL"),
        ]
        regions = _merge_segments(segs, max_gap=10)
        assert len(regions) == 2

    def test_unsorted_input(self):
        segs = [
            Segment(onset=50, offset=60, label="FEM"),
            Segment(onset=10, offset=20, label="MAL"),
        ]
        regions = _merge_segments(segs, max_gap=5)
        assert len(regions) == 2
        assert regions[0][0] == 10


# ---------------------------------------------------------------------------
# build_clips — basic (now VTC-based)
# ---------------------------------------------------------------------------


class TestBuildClips:
    def test_empty_segments(self):
        clips = build_clips([], file_duration=1000)
        assert clips == []

    def test_single_short_segment(self):
        vtc = [Segment(onset=100, offset=130, label="FEM")]
        clips = build_clips(vtc, file_duration=3600, buffer_s=5)
        assert len(clips) == 1
        assert clips[0].abs_onset == 95.0
        assert clips[0].abs_offset == 135.0
        assert len(clips[0].vtc_segments) == 1

    def test_vad_assigned_to_clip(self):
        """VAD segments overlapping the clip get assigned."""
        vtc = [Segment(onset=100, offset=130, label="FEM")]
        vad = [
            Segment(onset=95, offset=135),   # overlaps
            Segment(onset=500, offset=520),   # outside
        ]
        clips = build_clips(vtc, vad_segments=vad, file_duration=3600, buffer_s=5)
        assert len(clips) == 1
        assert len(clips[0].vad_segments) == 1
        assert len(clips[0].vtc_segments) == 1

    def test_buffer_clamped_to_zero(self):
        vtc = [Segment(onset=2, offset=30, label="FEM")]
        clips = build_clips(vtc, file_duration=3600, buffer_s=5)
        assert clips[0].abs_onset == 0.0

    def test_buffer_clamped_to_file_end(self):
        vtc = [Segment(onset=3590, offset=3598, label="MAL")]
        clips = build_clips(vtc, file_duration=3600, buffer_s=5)
        assert clips[0].abs_offset == 3600.0

    def test_segments_cluster_within_max_gap(self):
        """Two VTC segments 5s apart with max_gap=10 → one clip."""
        vtc = [
            Segment(onset=100, offset=200, label="FEM"),
            Segment(onset=205, offset=300, label="MAL"),
        ]
        clips = build_clips(vtc, file_duration=3600, max_gap=10, buffer_s=5)
        assert len(clips) == 1

    def test_segments_split_beyond_max_gap(self):
        """Two VTC segments far apart → separate clips."""
        vtc = [
            Segment(onset=100, offset=200, label="FEM"),
            Segment(onset=900, offset=1000, label="MAL"),
        ]
        clips = build_clips(vtc, file_duration=3600, max_gap=10, buffer_s=5)
        assert len(clips) == 2

    def test_min_seg_filter(self):
        """Segments shorter than min_seg_s are filtered out."""
        vtc = [
            Segment(onset=100, offset=100.3, label="KCHI"),  # 0.3s < 0.5s
            Segment(onset=200, offset=210, label="FEM"),     # 10s OK
        ]
        clips = build_clips(vtc, file_duration=3600, min_seg_s=0.5)
        assert len(clips) == 1
        assert len(clips[0].vtc_segments) == 1

    def test_all_short_segments_gives_no_clips(self):
        """If all VTC segments are filtered, no clips."""
        vtc = [
            Segment(onset=100, offset=100.3, label="KCHI"),
            Segment(onset=200, offset=200.2, label="OCH"),
        ]
        clips = build_clips(vtc, file_duration=3600, min_seg_s=0.5)
        assert clips == []

    def test_min_speech_filter(self):
        """Clip with < min_clip_speech_s of VTC speech is dropped."""
        vtc = [Segment(onset=100, offset=102, label="FEM")]  # 2s speech
        clips = build_clips(vtc, file_duration=3600, min_clip_speech_s=5, min_seg_s=0)
        assert len(clips) == 0

    def test_greedy_packing(self):
        """Three close VTC regions that fit into one clip → packed together."""
        vtc = [
            Segment(onset=100, offset=200, label="FEM"),
            Segment(onset=250, offset=300, label="MAL"),
            Segment(onset=350, offset=400, label="KCHI"),
        ]
        clips = build_clips(
            vtc, file_duration=3600,
            max_gap=5, buffer_s=5, max_clip_s=600,
        )
        # Three regions (gaps 50s > 5s), but total span 300s < 600s → 1 clip
        assert len(clips) == 1


# ---------------------------------------------------------------------------
# build_clips — splitting
# ---------------------------------------------------------------------------


class TestBuildClipsSplitting:
    def test_long_region_gets_split(self):
        """A 20-minute continuous VTC region → 2+ clips."""
        vtc = [Segment(onset=0, offset=1200, label="FEM")]
        clips = build_clips(
            vtc, file_duration=1200,
            max_clip_s=600, buffer_s=0, split_search_s=120, min_seg_s=0,
        )
        assert len(clips) >= 2
        for c in clips:
            assert c.duration <= 600 + 120

    def test_split_prefers_vtc_silence(self):
        """Insert a VTC gap at ~580s; the algorithm should split there."""
        vtc = [
            Segment(onset=0, offset=575, label="FEM"),
            Segment(onset=585, offset=1200, label="MAL"),
        ]
        clips = build_clips(
            vtc, file_duration=1200,
            max_clip_s=600, buffer_s=0, max_gap=5, split_search_s=120,
            min_seg_s=0,
        )
        assert len(clips) >= 2
        assert clips[0].abs_offset <= 600

    def test_buffer_does_not_remerge_splits(self):
        """Regression: buffer overlap must not cascade-merge all splits."""
        vtc = [Segment(onset=0, offset=7200, label="FEM")]
        clips = build_clips(
            vtc, file_duration=7200,
            max_clip_s=600, buffer_s=5, split_search_s=120, min_seg_s=0,
        )
        assert len(clips) >= 10
        for c in clips:
            assert c.duration <= 750

    def test_all_clips_respect_max_duration(self):
        """Every clip from dense VTC data stays near max_clip_s."""
        vtc = []
        for start in range(0, 50000, 20):
            vtc.append(Segment(onset=start, offset=start + 18, label="FEM"))
        clips = build_clips(
            vtc, file_duration=50000,
            max_clip_s=600, buffer_s=5, max_gap=10, split_search_s=120,
        )
        assert len(clips) >= 50
        for c in clips:
            assert c.duration <= 750


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
        # Check all new fields exist
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
        # intersection=10, union=30 → IoU ≈ 0.33
        assert 0.25 < iou < 0.40

    def test_empty_segments(self):
        iou = _compute_iou([], [], 0, 100)
        assert iou == 0.0
