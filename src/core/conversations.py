"""Turn and conversation detection from VTC segments.

Definitions
-----------
**Turn** — A contiguous block of VTC activity from a **single speaker
label**, bounded by either a gap >``min_gap_s`` of silence or a change of
label.  Only consecutive segments sharing the same label and separated by
≤``min_gap_s`` are merged into the same turn.  A different label always
starts a new turn, even with zero gap (simultaneous speech).  Each turn's
``label`` is the single VTC speaker class for that turn.

**Conversation** — A sequence of ≥1 turns where consecutive turns are
separated by <``max_silence_s``.  Conversations capture stretches of
active interaction; long silences (≥``max_silence_s``) break the
sequence.

**Transition** — Within a conversation, each consecutive pair of turns
whose dominant labels differ constitutes a speaker transition.  The gap
between those turns is the *response latency*.

These structures are used both for packaging metadata (per-clip) and for
dataset-level dashboard figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One conversational turn (merged VTC activity block)."""

    onset: float
    offset: float
    label: str  # dominant VTC label
    # Per-label durations within this turn
    label_durations: dict[str, float] = field(default_factory=dict)
    n_segments: int = 0

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class Conversation:
    """A sequence of turns separated by < max_silence_s."""

    onset: float
    offset: float
    turns: list[Turn] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def labels_present(self) -> list[str]:
        return sorted({t.label for t in self.turns})

    @property
    def is_multi_speaker(self) -> bool:
        return len(self.labels_present) > 1

    def transitions(self) -> list[tuple[str, str, float]]:
        """Return ``(from_label, to_label, gap_s)`` for each speaker change."""
        result: list[tuple[str, str, float]] = []
        for i in range(1, len(self.turns)):
            prev = self.turns[i - 1]
            curr = self.turns[i]
            if prev.label != curr.label:
                gap = curr.onset - prev.offset
                result.append((prev.label, curr.label, gap))
        return result

    def inter_turn_gaps(self) -> list[float]:
        """Gaps between consecutive turns within this conversation."""
        return [
            self.turns[i].onset - self.turns[i - 1].offset
            for i in range(1, len(self.turns))
        ]


@dataclass
class Transition:
    """A speaker-to-speaker handoff within a conversation."""

    from_label: str
    to_label: str
    gap_s: float  # time between end of from-turn and start of to-turn
    from_duration: float  # duration of the departing turn
    to_duration: float  # duration of the arriving turn


# ---------------------------------------------------------------------------
# Detection algorithms
# ---------------------------------------------------------------------------


def detect_turns(
    vtc_segments: list,
    min_gap_s: float = 0.3,
) -> list[Turn]:
    """Detect turns by merging same-label VTC segments within *min_gap_s*.

    A turn is a maximal run of VTC segments that share the same label and
    whose consecutive gaps are all ≤ *min_gap_s*.  A label change always
    breaks a turn, even when the two segments are adjacent or overlapping.

    Parameters
    ----------
    vtc_segments : list[Segment]
        VTC segments with ``.onset``, ``.offset``, ``.label`` attributes.
        Need not be sorted.
    min_gap_s : float
        Maximum silence gap (seconds) between same-label VTC segments that
        still counts as the same turn.  Gaps > min_gap_s start a new turn.

    Returns
    -------
    list[Turn]
        Sorted chronologically.  Each turn has a single ``label``
        (the speaker class of all its constituent segments).
    """
    if not vtc_segments:
        return []

    segs = sorted(vtc_segments, key=lambda s: s.onset)

    turns: list[Turn] = []
    # Accumulate segments for current turn
    group: list = [segs[0]]

    for seg in segs[1:]:
        prev_end = max(s.offset for s in group)
        same_label = seg.label == group[0].label
        within_gap = seg.onset - prev_end <= min_gap_s
        if same_label and within_gap:
            group.append(seg)
        else:
            # Label change or gap too large → flush current group as a turn
            turns.append(_group_to_turn(group))
            group = [seg]

    # Flush final group
    turns.append(_group_to_turn(group))
    return turns


def _group_to_turn(segments: list) -> Turn:
    """Convert a group of VTC segments into a Turn."""
    onset = min(s.onset for s in segments)
    offset = max(s.offset for s in segments)

    # Accumulate duration per label
    label_durs: dict[str, float] = {}
    for s in segments:
        lbl = s.label or "?"
        label_durs[lbl] = label_durs.get(lbl, 0.0) + s.duration

    dominant = max(label_durs, key=lambda k: label_durs[k])

    return Turn(
        onset=onset,
        offset=offset,
        label=dominant,
        label_durations=label_durs,
        n_segments=len(segments),
    )


def detect_conversations(
    turns: list[Turn],
    max_silence_s: float = 10.0,
) -> list[Conversation]:
    """Group turns into conversations.

    Parameters
    ----------
    turns : list[Turn]
        Sorted chronologically (output of :func:`detect_turns`).
    max_silence_s : float
        Maximum silence gap between consecutive turns that still keeps
        them in the same conversation.  Gaps ≥ max_silence_s start a
        new conversation.

    Returns
    -------
    list[Conversation]
        Sorted chronologically.
    """
    if not turns:
        return []

    conversations: list[Conversation] = []
    group: list[Turn] = [turns[0]]

    for turn in turns[1:]:
        gap = turn.onset - group[-1].offset
        if gap >= max_silence_s:
            conversations.append(_group_to_conversation(group))
            group = [turn]
        else:
            group.append(turn)

    conversations.append(_group_to_conversation(group))
    return conversations


def _group_to_conversation(turns: list[Turn]) -> Conversation:
    return Conversation(
        onset=turns[0].onset,
        offset=turns[-1].offset,
        turns=list(turns),
    )


def extract_transitions(conversations: list[Conversation]) -> list[Transition]:
    """Extract all speaker transitions across conversations."""
    result: list[Transition] = []
    for conv in conversations:
        for i in range(1, len(conv.turns)):
            prev = conv.turns[i - 1]
            curr = conv.turns[i]
            if prev.label != curr.label:
                result.append(Transition(
                    from_label=prev.label,
                    to_label=curr.label,
                    gap_s=curr.onset - prev.offset,
                    from_duration=prev.duration,
                    to_duration=curr.duration,
                ))
    return result


def inter_conversation_gaps(conversations: list[Conversation]) -> list[float]:
    """Return gaps (seconds) between consecutive conversations."""
    return [
        conversations[i].onset - conversations[i - 1].offset
        for i in range(1, len(conversations))
    ]


# ---------------------------------------------------------------------------
# Conversational bout detection
# ---------------------------------------------------------------------------


def detect_bouts(conversations: list[Conversation]) -> list[list[Turn]]:
    """Within each conversation, find alternating speaker *bouts*.

    A bout is a maximal sub-sequence of consecutive turns where the
    speaker alternates (A→B→A→B…).  A same-speaker repeat breaks the
    bout.

    Returns a flat list of bouts (each bout = list of Turn).
    """
    bouts: list[list[Turn]] = []
    for conv in conversations:
        if len(conv.turns) < 2:
            continue
        bout: list[Turn] = [conv.turns[0]]
        for t in conv.turns[1:]:
            if t.label != bout[-1].label:
                bout.append(t)
            else:
                if len(bout) >= 2:
                    bouts.append(bout)
                bout = [t]
        if len(bout) >= 2:
            bouts.append(bout)
    return bouts
