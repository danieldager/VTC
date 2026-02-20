"""Interval arithmetic — merging, IoU, and sample-index ↔ seconds conversion.

All functions here are pure (no I/O, no model dependency) and operate on
lists of ``(onset, offset)`` or ``(start_sample, end_sample, label)``
tuples.
"""

from __future__ import annotations

from segma.utils.conversions import frames_to_seconds


# ---------------------------------------------------------------------------
# Pair-level helpers  (seconds-based (onset, offset) tuples)
# ---------------------------------------------------------------------------


def merge_pairs(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping ``(onset, offset)`` intervals.

    Returns a sorted, non-overlapping list.
    """
    if not pairs:
        return []
    pairs = sorted(pairs)
    merged = [pairs[0]]
    for onset, offset in pairs[1:]:
        if onset <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], offset))
        else:
            merged.append((onset, offset))
    return merged


def total_duration(pairs: list[tuple[float, float]]) -> float:
    """Sum of durations across all ``(onset, offset)`` pairs."""
    return sum(b - a for a, b in pairs)


def compute_iou(
    vtc_pairs: list[tuple[float, float]],
    vad_pairs: list[tuple[float, float]],
) -> float:
    """Intersection-over-Union between two sets of merged ``(onset, offset)`` intervals."""
    if not vtc_pairs and not vad_pairs:
        return 1.0
    if not vtc_pairs or not vad_pairs:
        return 0.0

    tp = 0.0
    vi, ai = 0, 0
    while vi < len(vtc_pairs) and ai < len(vad_pairs):
        v_on, v_off = vtc_pairs[vi]
        a_on, a_off = vad_pairs[ai]
        overlap = min(v_off, a_off) - max(v_on, a_on)
        if overlap > 0:
            tp += overlap
        if v_off <= a_off:
            vi += 1
        else:
            ai += 1

    union = total_duration(vtc_pairs) + total_duration(vad_pairs) - tp
    return tp / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Sample-index intervals → seconds
# ---------------------------------------------------------------------------


def intervals_to_pairs(
    intervals: list[tuple[int, int, str]],
) -> list[tuple[float, float]]:
    """Convert sample-index intervals to merged ``(onset, offset)`` pairs in seconds."""
    pairs = []
    for start_f, end_f, _label in intervals:
        onset = round(float(frames_to_seconds(start_f)), 3)
        offset = round(float(frames_to_seconds(end_f)), 3)
        if offset > onset:
            pairs.append((onset, offset))
    return merge_pairs(sorted(pairs))


def intervals_to_segments(
    intervals: list[tuple[int, int, str]],
    uid: str,
) -> list[dict]:
    """Convert sample-index intervals to segment dicts.

    Each dict has keys: ``uid``, ``onset``, ``offset``, ``duration``, ``label``.
    """
    rows: list[dict] = []
    for start_f, end_f, label in intervals:
        onset = round(float(frames_to_seconds(start_f)), 3)
        offset = round(float(frames_to_seconds(end_f)), 3)
        duration = round(offset - onset, 3)
        if duration > 0:
            rows.append(
                {
                    "uid": uid,
                    "onset": onset,
                    "offset": offset,
                    "duration": duration,
                    "label": label,
                }
            )
    return rows
