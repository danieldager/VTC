"""Clip building — cluster VTC speech segments into ≤10-minute clips.

The algorithm uses **VTC segments** (labelled speech: FEM/MAL/KCHI/OCH) as
the primary signal for clip boundaries.  VAD segments are assigned to clips
as supplementary metadata for comparison.

Steps:

1. **Filter** VTC segments shorter than *min_seg_s* (removes coughs, etc.).
2. **Merge** nearby VTC segments within *max_gap* into contiguous regions.
3. **Greedily pack** small regions into clips up to *max_clip_s*.
4. **Split** regions longer than *max_clip_s* at the best VTC speaker
   boundary or silence within a ±\ *split_search_s* window.
5. **Buffer** clips with *buffer_s* of silence on each side, resolve
   overlaps without exceeding the clip limit.
6. **Assign** both VTC and VAD segments to each clip.
7. **Filter** clips with less than *min_clip_speech_s* of VTC speech.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A single continuous speech segment (absolute seconds)."""

    onset: float
    offset: float
    label: str | None = None  # VTC label (FEM / MAL / KCHI / OCH) or None for VAD

    @property
    def duration(self) -> float:
        return self.offset - self.onset


def _compute_iou(segs_a: list[Segment], segs_b: list[Segment],
                 clip_onset: float, clip_offset: float) -> float:
    """Compute frame-level IoU between two segment lists within a clip.

    Uses 0.1s resolution for efficiency.
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

    # Keep backward compat — "speech" now means VTC
    @property
    def speech_duration(self) -> float:
        return self.vtc_speech_duration

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
            self.vad_segments, self.vtc_segments,
            self.abs_onset, self.abs_offset,
        )

    def _vtc_gaps(self) -> list[float]:
        """Gaps between consecutive VTC segments (seconds)."""
        segs = sorted(self.vtc_segments, key=lambda s: s.onset)
        return [segs[i].onset - segs[i - 1].offset
                for i in range(1, len(segs))
                if segs[i].onset > segs[i - 1].offset]

    @property
    def mean_vtc_gap(self) -> float:
        gaps = self._vtc_gaps()
        return sum(gaps) / len(gaps) if gaps else 0.0

    @property
    def mean_vtc_seg_duration(self) -> float:
        if not self.vtc_segments:
            return 0.0
        return sum(s.duration for s in self.vtc_segments) / len(self.vtc_segments)

    def to_metadata(self, uid: str, clip_idx: int) -> dict:
        """Serialise to a JSON-friendly dict with **relative** timestamps."""
        return {
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
            # VAD stats
            "vad_speech_duration": round(self.vad_speech_duration, 3),
            "vad_speech_density": round(self.vad_density, 3),
            "n_vad_segments": len(self.vad_segments),
            # Agreement
            "vad_vtc_iou": round(self.vad_vtc_iou, 3),
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_segments(
    segs: list[Segment], max_gap: float
) -> list[tuple[float, float, list[Segment]]]:
    """Merge segments whose gap ≤ *max_gap* into contiguous regions.

    Returns ``[(region_onset, region_offset, [original_segments]), ...]``
    in chronological order.
    """
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s.onset)
    regions: list[tuple[float, float, list[Segment]]] = []
    cur_on, cur_off = segs[0].onset, segs[0].offset
    cur_segs: list[Segment] = [segs[0]]
    for s in segs[1:]:
        if s.onset - cur_off <= max_gap:
            cur_off = max(cur_off, s.offset)
            cur_segs.append(s)
        else:
            regions.append((cur_on, cur_off, cur_segs))
            cur_on, cur_off = s.onset, s.offset
            cur_segs = [s]
    regions.append((cur_on, cur_off, cur_segs))
    return regions


def _find_silence_gaps(
    segs: list[Segment],
    search_start: float,
    search_end: float,
) -> list[tuple[float, float]]:
    """Find silence gaps within [search_start, search_end].

    Returns ``[(gap_onset, gap_offset), ...]`` sorted by gap duration
    (longest first).
    """
    relevant = sorted(
        [s for s in segs if s.offset > search_start and s.onset < search_end],
        key=lambda s: s.onset,
    )
    if not relevant:
        return [(search_start, search_end)]

    gaps: list[tuple[float, float]] = []
    if relevant[0].onset > search_start:
        gaps.append((search_start, relevant[0].onset))
    for i in range(len(relevant) - 1):
        g_on = relevant[i].offset
        g_off = relevant[i + 1].onset
        if g_off > g_on:
            gaps.append((g_on, g_off))
    if relevant[-1].offset < search_end:
        gaps.append((relevant[-1].offset, search_end))

    return sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)


def _vtc_speaker_boundaries(
    vtc_segs: list[Segment],
    search_start: float,
    search_end: float,
) -> list[float]:
    """Find points where a VTC speaker segment ends within the window.

    Returns timestamps sorted by proximity to the window center (best
    split points first).
    """
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


def _split_long_region(
    region_on: float,
    region_off: float,
    vtc_segs: list[Segment],
    max_clip_s: float,
    split_search_s: float,
) -> list[tuple[float, float]]:
    """Split a region longer than *max_clip_s* into sub-clip boundaries.

    Strategy (uses VTC segments for both split heuristics):
      1. Prefer VTC silence gap midpoint (longest gap in search window).
      2. Then VTC speaker boundary (closest to window center).
      3. Hard cut at max_clip_s.
    """
    clips: list[tuple[float, float]] = []
    pos = region_on

    while pos < region_off:
        remaining = region_off - pos
        if remaining <= max_clip_s + split_search_s:
            clips.append((pos, region_off))
            break

        window_start = pos + max_clip_s - split_search_s
        window_end = min(pos + max_clip_s + split_search_s, region_off)

        # 1. Try VTC silence gap midpoint
        silence_gaps = _find_silence_gaps(vtc_segs, window_start, window_end)
        if silence_gaps:
            gap = silence_gaps[0]
            split_at = (gap[0] + gap[1]) / 2
            clips.append((pos, split_at))
            pos = split_at
            continue

        # 2. Try VTC speaker boundary
        boundaries = _vtc_speaker_boundaries(vtc_segs, window_start, window_end)
        if boundaries:
            split_at = boundaries[0]
            clips.append((pos, split_at))
            pos = split_at
            continue

        # 3. Hard cut
        split_at = pos + max_clip_s
        clips.append((pos, split_at))
        pos = split_at

    return clips


def build_clips(
    vtc_segments: list[Segment],
    vad_segments: list[Segment] | None = None,
    file_duration: float = 0.0,
    max_clip_s: float = 600.0,
    buffer_s: float = 5.0,
    max_gap: float = 10.0,
    split_search_s: float = 120.0,
    min_clip_speech_s: float = 5.0,
    min_seg_s: float = 0.5,
) -> list[Clip]:
    """Build clips from a single file's VTC + VAD segments.

    VTC segments drive the clip boundaries.  VAD segments are assigned
    to clips as supplementary metadata.

    Parameters
    ----------
    vtc_segments : list[Segment]
        VTC labelled segments (absolute seconds).  **Primary signal.**
    vad_segments : list[Segment] | None
        VAD speech segments (absolute seconds).  Optional — for comparison.
    file_duration : float
        Total audio file duration in seconds.
    max_clip_s : float
        Maximum clip duration (default 600 = 10 minutes).
    buffer_s : float
        Non-speech padding on each side of a clip.
    max_gap : float
        Maximum silence gap between VTC segments before they're split
        into separate regions (default 10s).
    split_search_s : float
        How far before/after the max_clip_s boundary to search for
        a good split point (default 120 = 2 minutes).
    min_clip_speech_s : float
        Discard clips with less VTC speech than this.
    min_seg_s : float
        Filter out VTC segments shorter than this (removes coughs,
        vocalizations, etc.).  Default 0.5s.

    Returns
    -------
    list[Clip]
        Clips sorted by abs_onset, each with VTC and VAD segments attached.
    """
    if vad_segments is None:
        vad_segments = []

    # --- Step 0: Filter short VTC segments ---
    vtc_filtered = [s for s in vtc_segments if s.duration >= min_seg_s]

    if not vtc_filtered:
        return []

    # --- Step 1: Merge VTC segments into regions ---
    regions = _merge_segments(vtc_filtered, max_gap)

    # --- Step 2: Greedy packing into clips ---
    raw_clips: list[tuple[float, float]] = []

    i = 0
    while i < len(regions):
        r_on, r_off, _ = regions[i]
        region_dur = r_off - r_on

        if region_dur > max_clip_s:
            sub_clips = _split_long_region(
                r_on, r_off, vtc_filtered,
                max_clip_s, split_search_s,
            )
            raw_clips.extend(sub_clips)
            i += 1
            continue

        # Try to pack subsequent regions into the same clip
        clip_on = r_on
        clip_off = r_off
        j = i + 1
        while j < len(regions):
            next_on, next_off, _ = regions[j]
            if next_off - clip_on > max_clip_s:
                break
            clip_off = next_off
            j += 1

        raw_clips.append((clip_on, clip_off))
        i = j

    # --- Step 3: Add buffer, clamp to file bounds ---
    buffered: list[tuple[float, float]] = []
    for c_on, c_off in raw_clips:
        b_on = max(0.0, c_on - buffer_s)
        b_off = min(file_duration, c_off + buffer_s) if file_duration > 0 else c_off + buffer_s
        buffered.append((b_on, b_off))

    # --- Step 4: Merge overlapping clips (buffers may cause overlap) ---
    buffered.sort()
    merged: list[tuple[float, float]] = [buffered[0]] if buffered else []
    for c_on, c_off in buffered[1:]:
        prev_on, prev_off = merged[-1]
        if c_on < prev_off:
            combined_off = max(prev_off, c_off)
            if combined_off - prev_on <= max_clip_s:
                merged[-1] = (prev_on, combined_off)
            else:
                mid = (c_on + prev_off) / 2
                merged[-1] = (prev_on, mid)
                merged.append((mid, c_off))
        else:
            merged.append((c_on, c_off))

    # --- Step 5: Assign segments to clips ---
    clips: list[Clip] = []
    for c_on, c_off in merged:
        clip = Clip(abs_onset=c_on, abs_offset=c_off)
        for s in vad_segments:
            if s.offset > c_on and s.onset < c_off:
                clip.vad_segments.append(Segment(
                    onset=max(s.onset, c_on),
                    offset=min(s.offset, c_off),
                ))
        for s in vtc_filtered:
            if s.offset > c_on and s.onset < c_off:
                clip.vtc_segments.append(Segment(
                    onset=max(s.onset, c_on),
                    offset=min(s.offset, c_off),
                    label=s.label,
                ))
        clips.append(clip)

    # --- Step 6: Filter out clips with negligible speech ---
    clips = [c for c in clips if c.vtc_speech_duration >= min_clip_speech_s]

    return clips
