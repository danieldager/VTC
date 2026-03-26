"""Build intermediate DataFrames from packaged clips for plotting / caching.

Every function returns a ``polars.DataFrame`` that can be saved as
parquet and later reloaded for the dashboard without re-running the
packaging pipeline.

DataFrames produced
-------------------
``build_clip_stats``
    One row per clip — duration, densities, IoU, SNR, C50, turn/conv counts.

``build_segment_stats``
    One row per VTC segment — onset, offset, label, SNR during segment.

``build_turn_stats``
    One row per conversational turn.

``build_conversation_stats``
    One row per conversation — duration, #turns, speaker labels, mean SNR/C50.

``build_transition_stats``
    One row per speaker transition — from/to labels, gap, durations.

``build_file_stats``
    One row per source file — aggregated clip-level data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import json

import polars as pl

from src.core import VTC_LABELS
from src.core.conversations import (
    Conversation,
    Turn,
    detect_conversations,
    detect_turns,
    extract_transitions,
    inter_conversation_gaps,
)
from src.packaging.clips import CUT_TIERS, Clip, Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_segment_snr(output_dir: Path) -> pl.DataFrame | None:
    """Load per-VTC-segment SNR/C50 parquets from ``output_dir/segment_snr/``.

    Returns a concatenated DataFrame with columns
    ``uid, onset, offset, label, snr_mean, c50_mean``, or *None* if
    the directory doesn't exist or is empty.
    """
    seg_snr_dir = output_dir / "segment_snr"
    if not seg_snr_dir.exists():
        return None
    files = sorted(seg_snr_dir.glob("shard_*.parquet"))
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files])
    if df.is_empty():
        return None
    loaded = len(files)
    n_uids = df["uid"].n_unique()
    print(f"  Loaded segment_snr: {loaded} shard(s), {n_uids} files, {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Clip-level stats
# ---------------------------------------------------------------------------


def build_clip_stats(
    clips: list[tuple[str, Path, int, Clip]],
    tier_counts: dict[str, int] | None = None,
    segment_stats_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per clip with all scalar metrics.

    When *segment_stats_df* is provided, SNR/C50 columns are aggregated
    from the per-segment values rather than from clip-level arrays.
    """
    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        # Conversation analysis per clip
        turns = detect_turns(clip.vtc_segments)
        convs = detect_conversations(turns)
        row = {
            "uid": uid,
            "clip_idx": clip_idx,
            "clip_id": f"{uid}_{clip_idx:04d}",
            "duration": round(clip.duration, 3),
            "abs_onset": round(clip.abs_onset, 3),
            "abs_offset": round(clip.abs_offset, 3),
            # VTC
            "vtc_speech_dur": round(clip.vtc_speech_duration, 3),
            "vtc_density": round(clip.speech_density, 3),
            "n_vtc_segments": len(clip.vtc_segments),
            "mean_vtc_seg_dur": round(clip.mean_vtc_seg_duration, 3),
            "mean_vtc_gap": round(clip.mean_vtc_gap, 3),
            "n_turns_raw": clip.n_turns,
            "n_labels": clip.n_labels,
            "has_adult": clip.has_adult,
            "dominant_label": clip.dominant_label or "?",
            "child_fraction": round(clip.child_fraction, 3),
            "child_speech_dur": round(clip.child_speech_duration, 3),
            "adult_speech_dur": round(clip.adult_speech_duration, 3),
            # VAD
            "vad_speech_dur": round(clip.vad_speech_duration, 3),
            "vad_density": round(clip.vad_density, 3),
            "n_vad_segments": len(clip.vad_segments),
            # Agreement
            "vad_vtc_iou": round(clip.vad_vtc_iou, 3),
            # Per-label durations
            **{
                f"dur_{l}": round(clip.label_durations.get(l, 0.0), 3)
                for l in VTC_LABELS
            },
            # Conversations
            "n_conv_turns": len(turns),
            "n_conversations": len(convs),
            "n_multi_speaker_convs": sum(1 for c in convs if c.is_multi_speaker),
            "turn_density_per_min": round(
                len(turns) / (clip.duration / 60) if clip.duration > 0 else 0, 2
            ),
            # Noise classification (from PANNs)
            "dominant_noise": clip.dominant_noise or "?",
        }
        # Add per-category mean probabilities
        profile = clip.noise_profile
        if profile:
            for cat, prob in profile.items():
                row[f"noise_{cat}"] = round(prob, 4)
        rows.append(row)
    df = pl.DataFrame(rows)

    # Derive SNR/C50 from segment-level stats
    snr_cols = [
        "snr_mean",
        "snr_std",
        "snr_min",
        "snr_max",
        "c50_mean",
        "c50_std",
        "c50_min",
        "c50_max",
    ]
    if segment_stats_df is not None and not segment_stats_df.is_empty():
        snr_agg = segment_stats_df.group_by(["uid", "clip_idx"]).agg(
            pl.col("snr_during").mean().round(1).alias("snr_mean"),
            pl.col("snr_during").std().round(1).alias("snr_std"),
            pl.col("snr_during").min().round(1).alias("snr_min"),
            pl.col("snr_during").max().round(1).alias("snr_max"),
            pl.col("c50_during").mean().round(1).alias("c50_mean"),
            pl.col("c50_during").std().round(1).alias("c50_std"),
            pl.col("c50_during").min().round(1).alias("c50_min"),
            pl.col("c50_during").max().round(1).alias("c50_max"),
        )
        df = df.join(snr_agg, on=["uid", "clip_idx"], how="left")
    else:
        for col in snr_cols:
            df = df.with_columns(pl.lit(None).alias(col).cast(pl.Float64))

    return df


# ---------------------------------------------------------------------------
# Segment-level stats
# ---------------------------------------------------------------------------


def build_segment_stats(
    clips: list[tuple[str, Path, int, Clip]],
    segment_snr_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per VTC segment — label, duration, SNR during segment.

    When *segment_snr_df* is provided (from ``segment_snr/`` parquet),
    SNR and C50 are joined by ``(uid, onset, offset)``.
    """
    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        for seg in clip.vtc_segments:
            rows.append(
                {
                    "uid": uid,
                    "clip_idx": clip_idx,
                    "onset": round(seg.onset, 3),
                    "offset": round(seg.offset, 3),
                    "duration": round(seg.duration, 3),
                    "label": seg.label or "?",
                }
            )
    df = pl.DataFrame(rows)
    if df.is_empty():
        return df.with_columns(
            pl.lit(None).alias("snr_during").cast(pl.Float64),
            pl.lit(None).alias("c50_during").cast(pl.Float64),
        )

    if segment_snr_df is not None and not segment_snr_df.is_empty():
        # Round join keys to match
        snr = segment_snr_df.select(
            pl.col("uid"),
            pl.col("onset").round(3),
            pl.col("offset").round(3),
            pl.col("snr_mean").round(1).alias("snr_during"),
            pl.col("c50_mean").round(1).alias("c50_during"),
        )
        df = df.join(snr, on=["uid", "onset", "offset"], how="left")
    else:
        df = df.with_columns(
            pl.lit(None).alias("snr_during").cast(pl.Float64),
            pl.lit(None).alias("c50_during").cast(pl.Float64),
        )
    return df


# ---------------------------------------------------------------------------
# Turn-level stats
# ---------------------------------------------------------------------------


def build_turn_stats(
    clips: list[tuple[str, Path, int, Clip]],
    min_gap_s: float = 0.3,
    segment_snr_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per conversational turn."""
    # Build a lookup dict: (uid, onset, offset) → (snr, c50)
    snr_lookup: dict[tuple[str, float, float], tuple[float | None, float | None]] = {}
    if segment_snr_df is not None:
        for row in segment_snr_df.iter_rows(named=True):
            key = (row["uid"], round(row["onset"], 3), round(row["offset"], 3))
            snr_lookup[key] = (row.get("snr_mean"), row.get("c50_mean"))

    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        turns = detect_turns(clip.vtc_segments, min_gap_s=min_gap_s)
        for turn_idx, turn in enumerate(turns):
            # Aggregate SNR/C50 from VTC segments within this turn
            snr_vals, c50_vals = [], []
            for seg in clip.vtc_segments:
                if seg.onset >= turn.onset and seg.offset <= turn.offset:
                    key = (uid, round(seg.onset, 3), round(seg.offset, 3))
                    s, c = snr_lookup.get(key, (None, None))
                    if s is not None:
                        snr_vals.append(s)
                    if c is not None:
                        c50_vals.append(c)
            snr_val = round(float(np.mean(snr_vals)), 1) if snr_vals else None
            c50_val = round(float(np.mean(c50_vals)), 1) if c50_vals else None
            rows.append(
                {
                    "uid": uid,
                    "clip_idx": clip_idx,
                    "turn_idx": turn_idx,
                    "onset": round(turn.onset, 3),
                    "offset": round(turn.offset, 3),
                    "duration": round(turn.duration, 3),
                    "label": turn.label,
                    "n_segments": turn.n_segments,
                    "snr_during": snr_val,
                    "c50_during": c50_val,
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conversation-level stats
# ---------------------------------------------------------------------------


def build_conversation_stats(
    clips: list[tuple[str, Path, int, Clip]],
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
    segment_snr_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per conversation — duration, turns, labels, SNR/C50."""
    # Build lookup: (uid, onset, offset) → (snr, c50)
    snr_lookup: dict[tuple[str, float, float], tuple[float | None, float | None]] = {}
    if segment_snr_df is not None:
        for row in segment_snr_df.iter_rows(named=True):
            key = (row["uid"], round(row["onset"], 3), round(row["offset"], 3))
            snr_lookup[key] = (row.get("snr_mean"), row.get("c50_mean"))

    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        turns = detect_turns(clip.vtc_segments, min_gap_s=min_gap_s)
        convs = detect_conversations(turns, max_silence_s=max_silence_s)
        ic_gaps = inter_conversation_gaps(convs)

        for conv_idx, conv in enumerate(convs):
            # Aggregate SNR/C50 from VTC segments within this conversation
            snr_vals, c50_vals = [], []
            for seg in clip.vtc_segments:
                if seg.onset >= conv.onset and seg.offset <= conv.offset:
                    key = (uid, round(seg.onset, 3), round(seg.offset, 3))
                    s, c = snr_lookup.get(key, (None, None))
                    if s is not None:
                        snr_vals.append(s)
                    if c is not None:
                        c50_vals.append(c)
            snr_val = round(float(np.mean(snr_vals)), 1) if snr_vals else None
            c50_val = round(float(np.mean(c50_vals)), 1) if c50_vals else None
            rows.append(
                {
                    "uid": uid,
                    "clip_idx": clip_idx,
                    "conv_idx": conv_idx,
                    "onset": round(conv.onset, 3),
                    "offset": round(conv.offset, 3),
                    "duration": round(conv.duration, 3),
                    "n_turns": conv.n_turns,
                    "is_multi_speaker": conv.is_multi_speaker,
                    "labels": ";".join(conv.labels_present),
                    "n_transitions": len(conv.transitions()),
                    "snr_mean": snr_val,
                    "c50_mean": c50_val,
                    "gap_after": (
                        round(ic_gaps[conv_idx], 3) if conv_idx < len(ic_gaps) else None
                    ),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transition-level stats
# ---------------------------------------------------------------------------


def build_transition_stats(
    clips: list[tuple[str, Path, int, Clip]],
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
) -> pl.DataFrame:
    """One row per speaker transition — from/to labels, gap, durations."""
    rows: list[dict] = []
    for uid, _audio_path, clip_idx, clip in clips:
        turns = detect_turns(clip.vtc_segments, min_gap_s=min_gap_s)
        convs = detect_conversations(turns, max_silence_s=max_silence_s)
        transitions = extract_transitions(convs)
        for tr in transitions:
            rows.append(
                {
                    "uid": uid,
                    "clip_idx": clip_idx,
                    "from_label": tr.from_label,
                    "to_label": tr.to_label,
                    "gap_s": round(tr.gap_s, 3),
                    "from_duration": round(tr.from_duration, 3),
                    "to_duration": round(tr.to_duration, 3),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# File-level stats
# ---------------------------------------------------------------------------


def build_file_stats(
    clips: list[tuple[str, Path, int, Clip]],
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
    segment_snr_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per source file — aggregated from its clips."""
    # Group clips by uid
    from collections import defaultdict

    by_uid: dict[str, list[Clip]] = defaultdict(list)
    for uid, _audio_path, _clip_idx, clip in clips:
        by_uid[uid].append(clip)

    # Pre-aggregate per-file SNR/C50 from segment data
    file_snr: dict[str, tuple[float | None, float | None]] = {}
    if segment_snr_df is not None and not segment_snr_df.is_empty():
        agg = segment_snr_df.group_by("uid").agg(
            pl.col("snr_mean").mean().round(1).alias("snr"),
            pl.col("c50_mean").mean().round(1).alias("c50"),
        )
        for row in agg.iter_rows(named=True):
            file_snr[row["uid"]] = (row["snr"], row["c50"])

    rows: list[dict] = []
    for uid, file_clips in by_uid.items():
        total_dur = sum(c.duration for c in file_clips)
        total_vtc = sum(c.vtc_speech_duration for c in file_clips)
        total_vad = sum(c.vad_speech_duration for c in file_clips)

        # Per-label total speech
        label_totals: dict[str, float] = {l: 0.0 for l in VTC_LABELS}
        for c in file_clips:
            for l, d in c.label_durations.items():
                if l in label_totals:
                    label_totals[l] += d

        # SNR / C50 from segment_snr_df
        snr_val, c50_val = file_snr.get(uid, (None, None))

        # Conversations across all clips for this file
        all_turns: list[Turn] = []
        all_convs: list[Conversation] = []
        for c in file_clips:
            turns = detect_turns(c.vtc_segments, min_gap_s=min_gap_s)
            convs = detect_conversations(turns, max_silence_s=max_silence_s)
            all_turns.extend(turns)
            all_convs.extend(convs)

        turn_durs = [t.duration for t in all_turns]
        conv_durs = [c.duration for c in all_convs]
        conv_turns = [c.n_turns for c in all_convs]

        rows.append(
            {
                "uid": uid,
                "n_clips": len(file_clips),
                "total_dur": round(total_dur, 3),
                "total_vtc_speech": round(total_vtc, 3),
                "total_vad_speech": round(total_vad, 3),
                "vtc_density": round(total_vtc / total_dur, 3) if total_dur > 0 else 0,
                **{f"total_dur_{l}": round(v, 3) for l, v in label_totals.items()},
                "snr_mean": snr_val,
                "c50_mean": c50_val,
                "n_turns": len(all_turns),
                "n_conversations": len(all_convs),
                "mean_turn_dur": (
                    round(float(np.mean(turn_durs)), 3) if turn_durs else 0.0
                ),
                "mean_conv_dur": (
                    round(float(np.mean(conv_durs)), 3) if conv_durs else 0.0
                ),
                "mean_turns_per_conv": (
                    round(float(np.mean(conv_turns)), 2) if conv_turns else 0.0
                ),
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save all DataFrames
# ---------------------------------------------------------------------------


def save_all_stats(
    clips: list[tuple[str, Path, int, Clip]],
    output_dir: Path,
    tier_counts: dict[str, int] | None = None,
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
) -> dict[str, pl.DataFrame]:
    """Build and save all intermediate DataFrames.

    Writes parquet files to ``output_dir/stats/`` and returns a dict
    of DataFrames keyed by name.
    """
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("Building intermediate DataFrames ...", flush=True)

    # Load per-VTC-segment SNR/C50 if available
    segment_snr_df = _load_segment_snr(output_dir)

    dfs: dict[str, pl.DataFrame] = {}

    # Segment stats first (needed by clip stats for SNR aggregation)
    dfs["segment_stats"] = build_segment_stats(clips, segment_snr_df)
    dfs["segment_stats"].write_parquet(stats_dir / "segment_stats.parquet")
    print(f"  segment_stats   : {len(dfs['segment_stats']):,} rows")

    dfs["clip_stats"] = build_clip_stats(
        clips,
        tier_counts,
        segment_stats_df=dfs["segment_stats"],
    )
    dfs["clip_stats"].write_parquet(stats_dir / "clip_stats.parquet")
    print(f"  clip_stats      : {len(dfs['clip_stats']):,} rows")

    dfs["turn_stats"] = build_turn_stats(
        clips,
        min_gap_s=min_gap_s,
        segment_snr_df=segment_snr_df,
    )
    dfs["turn_stats"].write_parquet(stats_dir / "turn_stats.parquet")
    print(f"  turn_stats      : {len(dfs['turn_stats']):,} rows")

    dfs["conversation_stats"] = build_conversation_stats(
        clips,
        min_gap_s=min_gap_s,
        max_silence_s=max_silence_s,
        segment_snr_df=segment_snr_df,
    )
    dfs["conversation_stats"].write_parquet(stats_dir / "conversation_stats.parquet")
    print(f"  conversation_stats: {len(dfs['conversation_stats']):,} rows")

    dfs["transition_stats"] = build_transition_stats(
        clips,
        min_gap_s=min_gap_s,
        max_silence_s=max_silence_s,
    )
    dfs["transition_stats"].write_parquet(stats_dir / "transition_stats.parquet")
    print(f"  transition_stats: {len(dfs['transition_stats']):,} rows")

    dfs["file_stats"] = build_file_stats(
        clips,
        min_gap_s=min_gap_s,
        max_silence_s=max_silence_s,
        segment_snr_df=segment_snr_df,
    )
    dfs["file_stats"].write_parquet(stats_dir / "file_stats.parquet")
    print(f"  file_stats      : {len(dfs['file_stats']):,} rows")

    # Correlation matrix across numeric clip-level columns
    numeric_cols = [
        "duration",
        "vtc_density",
        "vad_density",
        "vad_vtc_iou",
        "n_conv_turns",
        "n_labels",
        "child_fraction",
        "snr_mean",
        "snr_std",
        "c50_mean",
        "c50_std",
        "turn_density_per_min",
        "n_conversations",
    ]
    avail_cols = [c for c in numeric_cols if c in dfs["clip_stats"].columns]
    if avail_cols:
        corr_df = dfs["clip_stats"].select(avail_cols).to_pandas().corr()
        corr_pl = pl.from_pandas(
            corr_df.reset_index().rename(columns={"index": "metric"})
        )
        dfs["correlation"] = corr_pl
        corr_pl.write_parquet(stats_dir / "correlation_matrix.parquet")
        print(f"  correlation     : {len(avail_cols)}×{len(avail_cols)} matrix")

    if tier_counts is not None:
        (stats_dir / "tier_counts.json").write_text(json.dumps(tier_counts, indent=2))
        print(f"  tier_counts     : {sum(tier_counts.values()):,} cuts")

    print(f"  Saved to: {stats_dir}/")
    return dfs


# ---------------------------------------------------------------------------
# Load stats from disk (figures-only path)
# ---------------------------------------------------------------------------


def _snr_from_segment_stats(
    seg_df: pl.DataFrame,
) -> dict[tuple[str, float, float], tuple[float | None, float | None]]:
    """Build a lookup from segment_stats: (uid, onset, offset) → (snr, c50)."""
    lookup: dict[tuple[str, float, float], tuple[float | None, float | None]] = {}
    if "snr_during" not in seg_df.columns:
        return lookup
    for row in seg_df.iter_rows(named=True):
        key = (row["uid"], round(row["onset"], 3), round(row["offset"], 3))
        lookup[key] = (row.get("snr_during"), row.get("c50_during"))
    return lookup


def _recompute_turn_conv_stats(
    seg_df: pl.DataFrame,
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Rebuild turn, conversation, and transition DataFrames from segment_stats.

    SNR/C50 for turns and conversations are aggregated from the per-segment
    ``snr_during`` / ``c50_during`` columns already present in *seg_df*.
    """
    from src.packaging.clips import Segment  # avoid circular at module level

    snr_lookup = _snr_from_segment_stats(seg_df)

    turn_rows: list[dict] = []
    conv_rows: list[dict] = []
    trans_rows: list[dict] = []

    for (uid, clip_idx), clip_segs in seg_df.group_by(["uid", "clip_idx"]):
        segs = [
            Segment(onset=r["onset"], offset=r["offset"], label=r["label"])
            for r in clip_segs.iter_rows(named=True)
        ]
        turns = detect_turns(segs, min_gap_s=min_gap_s)
        convs = detect_conversations(turns, max_silence_s=max_silence_s)
        ic_gaps = inter_conversation_gaps(convs)
        transitions = extract_transitions(convs)

        for t_idx, t in enumerate(turns):
            snr_vals, c50_vals = [], []
            for seg in segs:
                if seg.onset >= t.onset and seg.offset <= t.offset:
                    key = (uid, round(seg.onset, 3), round(seg.offset, 3))
                    s, c = snr_lookup.get(key, (None, None))
                    if s is not None:
                        snr_vals.append(s)
                    if c is not None:
                        c50_vals.append(c)
            turn_rows.append(
                {
                    "uid": uid,
                    "clip_idx": clip_idx,
                    "turn_idx": t_idx,
                    "onset": round(t.onset, 3),
                    "offset": round(t.offset, 3),
                    "duration": round(t.duration, 3),
                    "label": t.label,
                    "n_segments": t.n_segments,
                    "snr_during": (
                        round(float(np.mean(snr_vals)), 1) if snr_vals else None
                    ),
                    "c50_during": (
                        round(float(np.mean(c50_vals)), 1) if c50_vals else None
                    ),
                }
            )

        for c_idx, conv in enumerate(convs):
            snr_vals, c50_vals = [], []
            for seg in segs:
                if seg.onset >= conv.onset and seg.offset <= conv.offset:
                    key = (uid, round(seg.onset, 3), round(seg.offset, 3))
                    s, c = snr_lookup.get(key, (None, None))
                    if s is not None:
                        snr_vals.append(s)
                    if c is not None:
                        c50_vals.append(c)
            conv_rows.append(
                {
                    "uid": uid,
                    "clip_idx": clip_idx,
                    "conv_idx": c_idx,
                    "onset": round(conv.onset, 3),
                    "offset": round(conv.offset, 3),
                    "duration": round(conv.duration, 3),
                    "n_turns": conv.n_turns,
                    "is_multi_speaker": conv.is_multi_speaker,
                    "labels": ";".join(conv.labels_present),
                    "n_transitions": len(conv.transitions()),
                    "snr_mean": (
                        round(float(np.mean(snr_vals)), 1) if snr_vals else None
                    ),
                    "c50_mean": (
                        round(float(np.mean(c50_vals)), 1) if c50_vals else None
                    ),
                    "gap_after": (
                        round(ic_gaps[c_idx], 3) if c_idx < len(ic_gaps) else None
                    ),
                }
            )

        for tr in transitions:
            trans_rows.append(
                {
                    "uid": uid,
                    "clip_idx": clip_idx,
                    "from_label": tr.from_label,
                    "to_label": tr.to_label,
                    "gap_s": round(tr.gap_s, 3),
                    "from_duration": round(tr.from_duration, 3),
                    "to_duration": round(tr.to_duration, 3),
                }
            )

    turn_df = pl.DataFrame(turn_rows) if turn_rows else pl.DataFrame()
    conv_df = pl.DataFrame(conv_rows) if conv_rows else pl.DataFrame()
    trans_df = pl.DataFrame(trans_rows) if trans_rows else pl.DataFrame()
    return turn_df, conv_df, trans_df


def load_all_stats(
    output_dir: Path,
    min_gap_s: float = 0.3,
    max_silence_s: float = 10.0,
) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    """Load cached stats parquets and recompute turn/conv/transition stats.

    Reads ``output_dir/stats/*.parquet`` for clip_stats, segment_stats,
    file_stats, and correlation.  Turn, conversation, and transition stats
    are recomputed from segment_stats using the current ``detect_turns``
    logic (so they reflect any recent changes to the turn definition).

    Returns
    -------
    dfs : dict[str, pl.DataFrame]
        Same keys as :func:`save_all_stats`.
    tier_counts : dict[str, int]
        Loaded from ``tier_counts.json`` if present, else empty dict.
    """
    stats_dir = output_dir / "stats"
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")

    dfs: dict[str, pl.DataFrame] = {}

    _load_map = {
        "clip_stats": "clip_stats",
        "segment_stats": "segment_stats",
        "file_stats": "file_stats",
        "correlation_matrix": "correlation",
    }
    for fname, key in _load_map.items():
        p = stats_dir / f"{fname}.parquet"
        if p.exists():
            dfs[key] = pl.read_parquet(p)
            print(f"  Loaded {fname}: {len(dfs[key]):,} rows")
        else:
            print(f"  WARN: {fname}.parquet not found, skipping")

    # Re-join fresh segment_snr if available (overwrites cached snr_during/c50_during)
    segment_snr_df = _load_segment_snr(output_dir)
    if segment_snr_df is not None and "segment_stats" in dfs:
        snr = segment_snr_df.select(
            pl.col("uid"),
            pl.col("onset").round(3),
            pl.col("offset").round(3),
            pl.col("snr_mean").round(1).alias("snr_during"),
            pl.col("c50_mean").round(1).alias("c50_during"),
        )
        seg = dfs["segment_stats"].drop(["snr_during", "c50_during"], strict=False)
        dfs["segment_stats"] = seg.join(snr, on=["uid", "onset", "offset"], how="left")
        print(
            f"  Re-joined segment_snr: {dfs['segment_stats']['snr_during'].null_count()} nulls"
        )

    if "segment_stats" not in dfs:
        raise FileNotFoundError(
            f"segment_stats.parquet not found in {stats_dir} — cannot recompute turns"
        )

    print(
        "Recomputing turn/conversation/transition stats from segment_stats ...",
        flush=True,
    )

    turn_df, conv_df, trans_df = _recompute_turn_conv_stats(
        dfs["segment_stats"],
        min_gap_s=min_gap_s,
        max_silence_s=max_silence_s,
    )
    dfs["turn_stats"] = turn_df
    dfs["conversation_stats"] = conv_df
    dfs["transition_stats"] = trans_df
    print(f"  turn_stats      : {len(turn_df):,} rows")
    print(f"  conversation_stats: {len(conv_df):,} rows")
    print(f"  transition_stats: {len(trans_df):,} rows")

    # Persist the recomputed stats so they're available for next time
    turn_df.write_parquet(stats_dir / "turn_stats.parquet")
    conv_df.write_parquet(stats_dir / "conversation_stats.parquet")
    trans_df.write_parquet(stats_dir / "transition_stats.parquet")
    print(f"  Updated parquets saved to {stats_dir}/")

    tier_counts: dict[str, int] = {}
    tc_path = stats_dir / "tier_counts.json"
    if tc_path.exists():
        tier_counts = json.loads(tc_path.read_text())

    return dfs, tier_counts
