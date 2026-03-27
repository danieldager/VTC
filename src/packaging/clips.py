"""Clip building — tile full audio files into ≤10-minute clips.

ALL audio is preserved.  VAD and VTC segments are used only to find safe
cut points (never cutting during speech or vocalisation).

Algorithm
---------
1. **Plan** how many clips are needed: ``ceil(duration / max_clip_s)``.
2. **Ideal clip length** = ``duration / n_clips`` (distributes evenly).
3. **Walk** ideal boundaries adaptively. For each one:
   a. Build a search window ``[ideal_pos − search, ideal_pos + search]``
      clamped to ``[prev_boundary, prev_boundary + max_clip_s]`` so the
      hard limit is never exceeded.
   b. Select a cut point using a **6-tier fallback chain** (see
      ``CUT_TIERS`` and the ``_find_cut`` nested function):

      1. Long silence gap (≥ ``min_gap_s``) in VAD∪VTC union.
      2. Any silence gap in VAD∪VTC union.
      3. Gap in VAD-only mask (VTC still active).
      4. Gap in VTC-only mask (VAD still active).
      5. VTC speaker-change boundary (inside active audio).
      6. Hard cut — no gaps or boundaries at all.

      Within each tier the midpoint closest to the ideal position is
      chosen to minimise cumulative drift.
   c. Recompute ``ideal_step`` from the *remaining* audio after each
      cut to keep clip lengths as uniform as possible.
4. **Assign** VTC and VAD segments to each clip (clamped to boundaries).

No audio is discarded — every second ends up in exactly one clip.

The function returns both the clip list and a ``dict[str, int]`` of
tier usage counts (see ``CUT_TIERS``) so callers can report cut-quality
statistics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Segma chunk grid geometry (HuBERT surgical_hydra)
# 199 windows × 320-sample stride = 63,680 samples at 16 kHz = 3.98 s
_SR: int = 16_000
CHUNK_STEP_F: int = 63_680  # samples
CHUNK_STEP_S: float = CHUNK_STEP_F / _SR  # 3.98 s


def snap_to_grid(t: float, file_duration: float) -> float:
    """Round *t* to the nearest multiple of ``CHUNK_STEP_S``.

    The result is clamped to ``[0, file_duration]``.
    """
    snapped = round(t / CHUNK_STEP_S) * CHUNK_STEP_S
    return max(0.0, min(snapped, file_duration))


# Tier labels for cut-point statistics (order = priority)
CUT_TIERS: tuple[str, ...] = (
    "long_union_gap",  # 1. Long silence gap (≥ min_gap_s) in VAD∪VTC union
    "short_union_gap",  # 2. Any silence gap in VAD∪VTC union
    "vad_only_gap",  # 3. Gap in VAD-only mask (VTC still active)
    "vtc_only_gap",  # 4. Gap in VTC-only mask (VAD still active)
    "speaker_boundary",  # 5. VTC speaker-change boundary (inside active audio)
    "hard_cut",  # 6. No gaps or boundaries — forced cut
    "degenerate_window",  # 0. Degenerate search window — forced cut
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A single continuous speech/vocalisation segment (absolute seconds)."""

    onset: float
    offset: float
    label: str | None = None  # VTC label (FEM / MAL / KCHI / OCH) or None for VAD

    @property
    def duration(self) -> float:
        return self.offset - self.onset


def _compute_iou(
    segs_a: list[Segment], segs_b: list[Segment], clip_onset: float, clip_offset: float
) -> float:
    """Compute frame-level IoU between two segment lists within a clip.

    Uses 0.1 s resolution for efficiency.
    """
    dur = clip_offset - clip_onset
    if dur <= 0:
        return 0.0
    step = 0.1
    n = max(1, int(dur / step))

    a_flags = bytearray(n)
    b_flags = bytearray(n)

    for s in segs_a:
        lo = max(0, int((s.onset - clip_onset) / step))
        hi = min(n, int((s.offset - clip_onset) / step))
        for i in range(lo, hi):
            a_flags[i] = 1

    for s in segs_b:
        lo = max(0, int((s.onset - clip_onset) / step))
        hi = min(n, int((s.offset - clip_onset) / step))
        for i in range(lo, hi):
            b_flags[i] = 1

    intersection = sum(a & b for a, b in zip(a_flags, b_flags))
    union = sum(a | b for a, b in zip(a_flags, b_flags))
    return intersection / union if union > 0 else 0.0


@dataclass
class Clip:
    """One packaged audio clip — at most *max_clip_s* of audio."""

    # Absolute time range within the source file
    abs_onset: float
    abs_offset: float
    # Segments (absolute coords — will be made relative at export)
    vad_segments: list[Segment] = field(default_factory=list)
    vtc_segments: list[Segment] = field(default_factory=list)
    # Generic feature store — loaders attach arbitrary per-clip data here.
    # Keys are feature names (e.g. "esc_array", "esc_categories").
    features: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def duration(self) -> float:
        return self.abs_offset - self.abs_onset

    @property
    def vtc_speech_duration(self) -> float:
        """Total VTC-labelled speech within this clip."""
        return sum(s.duration for s in self.vtc_segments)

    @property
    def vad_speech_duration(self) -> float:
        """Total VAD speech within this clip."""
        return sum(s.duration for s in self.vad_segments)

    @property
    def speech_density(self) -> float:
        dur = self.duration
        return self.vtc_speech_duration / dur if dur > 0 else 0.0

    @property
    def vad_density(self) -> float:
        dur = self.duration
        return self.vad_speech_duration / dur if dur > 0 else 0.0

    @property
    def n_turns(self) -> int:
        """Number of speaker turns (label changes between consecutive segments)."""
        if len(self.vtc_segments) <= 1:
            return len(self.vtc_segments)
        turns = 1
        for i in range(1, len(self.vtc_segments)):
            if self.vtc_segments[i].label != self.vtc_segments[i - 1].label:
                turns += 1
        return turns

    @property
    def labels_present(self) -> list[str]:
        """Unique VTC labels present in this clip, sorted."""
        return sorted({s.label for s in self.vtc_segments if s.label})

    @property
    def n_labels(self) -> int:
        return len(self.labels_present)

    @property
    def has_adult(self) -> bool:
        return any(s.label in ("FEM", "MAL") for s in self.vtc_segments)

    @property
    def vad_vtc_iou(self) -> float:
        """Frame-level IoU between VAD and VTC segments within this clip."""
        return _compute_iou(
            self.vad_segments,
            self.vtc_segments,
            self.abs_onset,
            self.abs_offset,
        )

    def _vtc_gaps(self) -> list[float]:
        """Gaps between consecutive VTC segments (seconds)."""
        segs = sorted(self.vtc_segments, key=lambda s: s.onset)
        return [
            segs[i].onset - segs[i - 1].offset
            for i in range(1, len(segs))
            if segs[i].onset > segs[i - 1].offset
        ]

    @property
    def mean_vtc_gap(self) -> float:
        gaps = self._vtc_gaps()
        return sum(gaps) / len(gaps) if gaps else 0.0

    @property
    def mean_vtc_seg_duration(self) -> float:
        if not self.vtc_segments:
            return 0.0
        return sum(s.duration for s in self.vtc_segments) / len(self.vtc_segments)

    # --- Noise properties (read from features dict) ---

    @property
    def esc_array(self) -> np.ndarray | None:
        return self.features.get("esc_array")

    @esc_array.setter
    def esc_array(self, value: np.ndarray | None) -> None:
        self.features["esc_array"] = value

    @property
    def esc_categories(self) -> list[str]:
        return self.features.get("esc_categories", [])

    @esc_categories.setter
    def esc_categories(self, value: list[str]) -> None:
        self.features["esc_categories"] = value

    @property
    def esc_step_s(self) -> float:
        return self.features.get("esc_step_s", 1.0)

    @esc_step_s.setter
    def esc_step_s(self, value: float) -> None:
        self.features["esc_step_s"] = value

    @property
    def esc_profile(self) -> dict[str, float] | None:
        """Mean probability per ESC category across the clip."""
        arr = self.esc_array
        if arr is None or len(arr) == 0:
            return None
        means = arr.mean(axis=0).astype(np.float32)
        return {cat: float(means[i]) for i, cat in enumerate(self.esc_categories)}

    @property
    def dominant_esc(self) -> str | None:
        """Noise category with highest mean probability in this clip."""
        profile = self.esc_profile
        if not profile:
            return None
        return max(profile, key=lambda k: profile[k])

    # --- Per-label properties ---

    @property
    def label_durations(self) -> dict[str, float]:
        """Total speech duration per VTC label."""
        ld: dict[str, float] = {}
        for s in self.vtc_segments:
            if s.label:
                ld[s.label] = ld.get(s.label, 0.0) + s.duration
        return ld

    @property
    def child_speech_duration(self) -> float:
        """Total KCHI + OCH speech in this clip."""
        ld = self.label_durations
        return ld.get("KCHI", 0.0) + ld.get("OCH", 0.0)

    @property
    def adult_speech_duration(self) -> float:
        """Total FEM + MAL speech in this clip."""
        ld = self.label_durations
        return ld.get("FEM", 0.0) + ld.get("MAL", 0.0)

    @property
    def child_fraction(self) -> float:
        """Fraction of VTC speech that is child (KCHI + OCH)."""
        total = self.vtc_speech_duration
        return self.child_speech_duration / total if total > 0 else 0.0

    @property
    def dominant_label(self) -> str | None:
        """VTC label with the most speech in this clip."""
        ld = self.label_durations
        return max(ld, key=lambda k: ld[k]) if ld else None

    def vad_coverage_by_label(self) -> dict[str, float]:
        """For each VTC label, fraction of that label's time covered by VAD."""
        result: dict[str, float] = {}
        for label in self.labels_present:
            label_segs = [s for s in self.vtc_segments if s.label == label]
            total_dur = sum(s.duration for s in label_segs)
            if total_dur == 0:
                continue
            covered = 0.0
            for ls in label_segs:
                for vs in self.vad_segments:
                    ov_start = max(ls.onset, vs.onset)
                    ov_end = min(ls.offset, vs.offset)
                    if ov_end > ov_start:
                        covered += ov_end - ov_start
            result[label] = min(covered / total_dur, 1.0)
        return result

    def to_metadata(self, uid: str, clip_idx: int) -> dict:
        """Serialise to a JSON-friendly dict with **relative** timestamps.

        Core fields are always present.  Feature loaders may add extra
        keys via the ``features`` dict — any JSON-serialisable value
        stored there under a key **not** already used by the core schema
        is merged into the output automatically.
        """
        meta: dict[str, Any] = {
            "uid": uid,
            "clip_idx": clip_idx,
            "clip_id": f"{uid}_{clip_idx:04d}",
            "abs_onset": round(self.abs_onset, 3),
            "abs_offset": round(self.abs_offset, 3),
            "duration": round(self.duration, 3),
            # VTC speech stats
            "vtc_speech_duration": round(self.vtc_speech_duration, 3),
            "vtc_speech_density": round(self.speech_density, 3),
            "n_vtc_segments": len(self.vtc_segments),
            "mean_vtc_seg_duration": round(self.mean_vtc_seg_duration, 3),
            "mean_vtc_gap": round(self.mean_vtc_gap, 3),
            "n_turns": self.n_turns,
            "n_labels": self.n_labels,
            "labels_present": self.labels_present,
            "has_adult": self.has_adult,
            # Per-label
            "child_speech_duration": round(self.child_speech_duration, 3),
            "adult_speech_duration": round(self.adult_speech_duration, 3),
            "child_fraction": round(self.child_fraction, 3),
            "dominant_label": self.dominant_label,
            "label_durations": {
                k: round(v, 3) for k, v in self.label_durations.items()
            },
            "vad_coverage_by_label": {
                k: round(v, 3) for k, v in self.vad_coverage_by_label().items()
            },
            # VAD stats
            "vad_speech_duration": round(self.vad_speech_duration, 3),
            "vad_speech_density": round(self.vad_density, 3),
            "n_vad_segments": len(self.vad_segments),
            # Agreement
            "vad_vtc_iou": round(self.vad_vtc_iou, 3),
            # Noise classification (from PANNs)
            "dominant_esc": self.dominant_esc,
            "esc_profile": (
                {k: round(v, 4) for k, v in self.esc_profile.items()}
                if self.esc_profile is not None
                else None
            ),
            # Segment details (relative timestamps)
            "vad_segments": [
                {
                    "onset": round(s.onset - self.abs_onset, 3),
                    "offset": round(s.offset - self.abs_onset, 3),
                    "duration": round(s.duration, 3),
                }
                for s in self.vad_segments
            ],
            "vtc_segments": [
                {
                    "onset": round(s.onset - self.abs_onset, 3),
                    "offset": round(s.offset - self.abs_onset, 3),
                    "duration": round(s.duration, 3),
                    "label": s.label,
                }
                for s in self.vtc_segments
            ],
        }
        # Merge extra features that loaders have attached.
        # Internal keys (numpy arrays, raw loader state) are skipped;
        # only JSON-serialisable scalars / dicts / lists are included.
        _INTERNAL = {"esc_array", "esc_categories", "esc_step_s"}
        for key, val in self.features.items():
            if key in _INTERNAL or key in meta:
                continue
            meta[key] = val
        return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_activity_union(
    vtc_segments: list[Segment],
    vad_segments: list[Segment],
) -> list[Segment]:
    """Merge all VAD + VTC segments into sorted, non-overlapping intervals.

    The result represents all times where audio is "active" (speech or
    vocalisation) — cutting here should be avoided.
    """
    all_segs = [(s.onset, s.offset) for s in vtc_segments] + [
        (s.onset, s.offset) for s in vad_segments
    ]
    if not all_segs:
        return []

    all_segs.sort()
    merged: list[tuple[float, float]] = [all_segs[0]]
    for on, off in all_segs[1:]:
        prev_on, prev_off = merged[-1]
        if on <= prev_off:
            merged[-1] = (prev_on, max(prev_off, off))
        else:
            merged.append((on, off))

    return [Segment(onset=on, offset=off) for on, off in merged]


def _find_silence_gaps(
    active: list[Segment],
    search_start: float,
    search_end: float,
) -> list[tuple[float, float]]:
    """Find silence gaps within [search_start, search_end].

    *active* must be sorted and non-overlapping (output of
    ``_build_activity_union``).

    Returns ``[(gap_onset, gap_offset), ...]`` sorted longest-first.
    """
    relevant = sorted(
        [s for s in active if s.offset > search_start and s.onset < search_end],
        key=lambda s: s.onset,
    )
    if not relevant:
        return [(search_start, search_end)]

    gaps: list[tuple[float, float]] = []
    # Gap before first active segment
    edge = max(relevant[0].onset, search_start)
    if edge > search_start:
        gaps.append((search_start, edge))
    # Gaps between active segments
    for i in range(len(relevant) - 1):
        g_on = max(relevant[i].offset, search_start)
        g_off = min(relevant[i + 1].onset, search_end)
        if g_off > g_on:
            gaps.append((g_on, g_off))
    # Gap after last active segment
    edge = min(relevant[-1].offset, search_end)
    if edge < search_end:
        gaps.append((edge, search_end))

    return sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)


def _vtc_speaker_boundaries(
    vtc_segs: list[Segment],
    search_start: float,
    search_end: float,
) -> list[float]:
    """VTC segment boundaries within the window, closest to centre first."""
    center = (search_start + search_end) / 2
    boundaries: list[float] = []
    for s in vtc_segs:
        if search_start < s.offset < search_end:
            boundaries.append(s.offset)
        if search_start < s.onset < search_end:
            boundaries.append(s.onset)
    return sorted(boundaries, key=lambda t: abs(t - center))


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def build_clips(
    vtc_segments: list[Segment],
    vad_segments: list[Segment] | None = None,
    file_duration: float = 0.0,
    max_clip_s: float = 600.0,
    split_search_s: float = 180.0, # size of the search window
    min_gap_s: float = 10.0,
    snap_to_chunk_grid: bool = True,
) -> tuple[list[Clip], dict[str, int]]:
    """Tile a full audio file into clips of roughly equal length.

    Every second of ``[0, file_duration]`` ends up in exactly one clip.
    The number of clips is ``ceil(file_duration / max_clip_s)``, so the
    ideal clip length is ``file_duration / n_clips``.  Cut points are
    placed in the longest silence gap within a search window around
    each ideal boundary.

    Parameters
    ----------
    vtc_segments : list[Segment]
        VTC labelled segments (absolute seconds).
    vad_segments : list[Segment] | None
        VAD speech segments (absolute seconds).  Optional.
    file_duration : float
        Total audio file duration in seconds.
    max_clip_s : float
        Hard maximum clip duration (default 600 = 10 minutes).
    split_search_s : float
        How far before/after the ideal boundary to search for a good
        cut point (default 120 = 2 minutes).
    min_gap_s : float
        Silence gaps ≥ this length are strongly preferred (default 10 s).

    snap_to_chunk_grid : bool
        When *True* (default), snap every interior cut boundary to the
        nearest multiple of the segma chunk step (3.98 s).  This ensures
        that the model's 4-second analysis windows align with those used
        during full-file inference, so VTC predictions on clips are
        bit-identical to full-file predictions.  The maximum shift is
        ±1.99 s.

    Returns
    -------
    (clips, tier_counts) : tuple[list[Clip], dict[str, int]]
        *clips* — clips covering ``[0, file_duration]``, sorted
        chronologically.  *tier_counts* — how many cuts used each
        fallback tier (see ``CUT_TIERS``).
    """
    tier_counts: dict[str, int] = {t: 0 for t in CUT_TIERS}

    if file_duration <= 0:
        return [], tier_counts
    if vad_segments is None:
        vad_segments = []

    # --- Build activity masks ---
    active = _build_activity_union(vtc_segments, vad_segments)
    vad_only = _build_activity_union([], vad_segments)
    vtc_only = _build_activity_union(vtc_segments, [])

    # --- Plan: how many clips? ---
    n_clips = max(1, math.ceil(file_duration / max_clip_s))

    if n_clips == 1:
        # Whole file fits in one clip
        clip = Clip(abs_onset=0.0, abs_offset=file_duration)
        for s in vad_segments:
            if s.offset > 0 and s.onset < file_duration:
                clip.vad_segments.append(
                    Segment(
                        onset=max(s.onset, 0.0),
                        offset=min(s.offset, file_duration),
                    )
                )
        for s in vtc_segments:
            if s.offset > 0 and s.onset < file_duration:
                clip.vtc_segments.append(
                    Segment(
                        onset=max(s.onset, 0.0),
                        offset=min(s.offset, file_duration),
                        label=s.label,
                    )
                )
        return [clip], tier_counts

    # --- Place cut boundaries ---
    boundaries: list[float] = [0.0]

    def _find_cut(prev: float, ideal_pos: float) -> tuple[float, str]:
        """Find best cut point near *ideal_pos*, respecting hard limit.

        Fallback chain
        --------------
        1. Long silence gap (≥ min_gap_s) in VAD∪VTC union → midpoint
           closest to ideal.
        2. Any silence gap in VAD∪VTC union → midpoint closest to ideal.
        3. Gap in VAD-only mask (VTC may be active, but no speech
           detected by VAD) → midpoint closest to ideal.
        4. Gap in VTC-only mask (VAD may be active, but no labelled
           speaker) → midpoint closest to ideal.
        5. VTC speaker boundary closest to ideal (may land inside
           active audio — logged as warning).
        6. Hard cut at ideal position — no gap at all (logged as
           warning).
        """
        window_start = max(ideal_pos - split_search_s, prev + 1.0)
        window_end = min(
            ideal_pos + split_search_s, prev + max_clip_s, file_duration - 1.0
        )

        if window_start >= window_end:
            cut = min(prev + max_clip_s, file_duration)
            log.warning(
                "degenerate window at %.1fs (prev=%.1f, ideal=%.1f) — " "forced cut",
                cut,
                prev,
                ideal_pos,
            )
            return cut, "degenerate_window"

        def _best_gap(gap_list):
            """Pick gap whose midpoint is closest to ideal."""
            if not gap_list:
                return None
            g = min(gap_list, key=lambda g: abs((g[0] + g[1]) / 2 - ideal_pos))
            return (g[0] + g[1]) / 2

        # 1–2. Silence gap in full union (prefer long, then any)
        gaps = _find_silence_gaps(active, window_start, window_end)
        long_gaps = [(on, off) for on, off in gaps if off - on >= min_gap_s]
        cut = _best_gap(long_gaps) if long_gaps else None
        if cut is not None:
            return cut, "long_union_gap"
        cut = _best_gap(gaps)
        if cut is not None:
            return cut, "short_union_gap"

        # 3. VAD-only gap (no VAD activity, even if VTC labels exist)
        vad_gaps = _find_silence_gaps(vad_only, window_start, window_end)
        cut = _best_gap(vad_gaps)
        if cut is not None:
            log.info(
                "cut at %.1fs in VAD-only gap (VTC still active)",
                cut,
            )
            return cut, "vad_only_gap"

        # 4. VTC-only gap (no VTC labels, even if VAD detects speech)
        vtc_gaps = _find_silence_gaps(vtc_only, window_start, window_end)
        cut = _best_gap(vtc_gaps)
        if cut is not None:
            log.info(
                "cut at %.1fs in VTC-only gap (VAD still active)",
                cut,
            )
            return cut, "vtc_only_gap"

        # 5. VTC speaker boundary closest to ideal position
        spk = _vtc_speaker_boundaries(vtc_segments, window_start, window_end)
        if spk:
            spk.sort(key=lambda t: abs(t - ideal_pos))
            cut = spk[0]
            log.warning(
                "cut at %.1fs at speaker boundary inside active audio "
                "[window %.1f–%.1f]",
                cut,
                window_start,
                window_end,
            )
            return cut, "speaker_boundary"

        # 6. Hard cut — no gaps, no boundaries at all
        cut = min(ideal_pos, prev + max_clip_s)
        log.warning(
            "hard cut at %.1fs — continuous activity, no gaps or "
            "boundaries in [%.1f–%.1f]",
            cut,
            window_start,
            window_end,
        )
        return cut, "hard_cut"

    # --- Place cut boundaries adaptively ---
    # Keep splitting until the remaining audio fits in one clip.
    # Each step recomputes ideal_step from remaining audio, so cuts
    # always target an even distribution of the remaining duration.
    boundaries: list[float] = [0.0]

    while file_duration - boundaries[-1] > max_clip_s:
        prev = boundaries[-1]
        remaining = file_duration - prev
        n_remaining = max(1, math.ceil(remaining / max_clip_s))
        ideal_step = remaining / n_remaining
        ideal_pos = prev + ideal_step
        cut, tier = _find_cut(prev, ideal_pos)
        if snap_to_chunk_grid:
            cut = snap_to_grid(cut, file_duration)
        tier_counts[tier] += 1
        boundaries.append(cut)

    # Deduplicate: snapping can cause two cuts to collapse to the same
    # grid point.  Keep only strictly increasing boundaries.
    if snap_to_chunk_grid:
        deduped: list[float] = [boundaries[0]]
        for b in boundaries[1:]:
            if b > deduped[-1]:
                deduped.append(b)
        boundaries = deduped

    boundaries.append(file_duration)

    # --- Build Clip objects and assign segments ---
    clips: list[Clip] = []
    for i in range(len(boundaries) - 1):
        c_on, c_off = boundaries[i], boundaries[i + 1]
        clip = Clip(abs_onset=c_on, abs_offset=c_off)

        for s in vad_segments:
            if s.offset > c_on and s.onset < c_off:
                clip.vad_segments.append(
                    Segment(
                        onset=max(s.onset, c_on),
                        offset=min(s.offset, c_off),
                    )
                )
        for s in vtc_segments:
            if s.offset > c_on and s.onset < c_off:
                clip.vtc_segments.append(
                    Segment(
                        onset=max(s.onset, c_on),
                        offset=min(s.offset, c_off),
                        label=s.label,
                    )
                )
        clips.append(clip)

    return clips, tier_counts
