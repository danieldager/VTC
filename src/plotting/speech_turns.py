"""Dashboard pages: Conversational Structure, Turns & Conversations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.core import LABEL_COLORS, VTC_LABELS


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------


def _merge_intervals(segs: np.ndarray) -> list:
    """Merge overlapping intervals. Input: (n, 2) float64 array sorted by onset."""
    if len(segs) == 0:
        return []
    merged = [[segs[0, 0], segs[0, 1]]]
    for onset, offset in segs[1:]:
        if onset <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], offset)
        else:
            merged.append([onset, offset])
    return merged


def _intersection_duration(ma: list, mb: list) -> float:
    """Total intersection length of two sorted non-overlapping interval lists."""
    total = 0.0
    i = j = 0
    while i < len(ma) and j < len(mb):
        lo = max(ma[i][0], mb[j][0])
        hi = min(ma[i][1], mb[j][1])
        if lo < hi:
            total += hi - lo
        if ma[i][1] < mb[j][1]:
            i += 1
        else:
            j += 1
    return total


def _compute_label_overlap_minutes(segment_df: pl.DataFrame) -> dict[str, float]:
    """Total simultaneous speech (minutes) between every pair of VTC labels.

    For each clip, the union of each label's segments is computed, then
    pairwise intersections are summed.
    """
    result: dict[str, float] = {}
    if len(segment_df) == 0:
        return result
    for (_, _), clip_segs in segment_df.group_by(["uid", "clip_idx"]):
        labels = sorted(clip_segs["label"].unique().to_list())
        if len(labels) < 2:
            continue
        intervals: dict[str, list] = {}
        for lbl in labels:
            raw = (
                clip_segs.filter(pl.col("label") == lbl)
                .select(["onset", "offset"])
                .sort("onset")
                .to_numpy()
            )
            intervals[lbl] = _merge_intervals(raw)
        for i, la in enumerate(labels):
            for lb in labels[i + 1 :]:
                ov = _intersection_duration(intervals[la], intervals[lb])
                key = f"{la}+{lb}"
                result[key] = result.get(key, 0.0) + ov
    return {k: v / 60 for k, v in result.items()}


def _setup():
    """Lazy matplotlib setup, returns plt module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_conversation_figures(
    clip_df: pl.DataFrame,
    turn_df: pl.DataFrame,
    conversation_df: pl.DataFrame,
    transition_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Conversational structure dashboard — 3×3 panels."""
    plt = _setup()

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Conversational Structure", fontsize=14, fontweight="bold")

    # Split: true conversations (≥2 speakers) vs monologues (1 speaker)
    conv_df = conversation_df.filter(pl.col("is_multi_speaker") == True)  # noqa: E712
    mono_df = conversation_df.filter(pl.col("is_multi_speaker") == False)  # noqa: E712

    # 1. Turn density histogram (turns per minute)
    ax = axes[0, 0]
    td = clip_df["turn_density_per_min"].drop_nulls().to_numpy()
    if len(td) > 0:
        ax.hist(td, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(td),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(td):.1f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Turns per minute")
    ax.set_ylabel("Count")
    ax.set_title("Turn Density per Clip")

    # 2. Turn type breakdown (stacked bar: who→who)
    ax = axes[0, 1]
    if len(transition_df) > 0:
        pairs = transition_df.group_by(["from_label", "to_label"]).len()
        pairs = pairs.sort("len", descending=True).head(12)
        pair_labels = [
            f"{r['from_label']}→{r['to_label']}" for r in pairs.iter_rows(named=True)
        ]
        pair_counts = pairs["len"].to_list()
        colors = plt.cm.tab20(np.linspace(0, 1, len(pair_labels)))  # type: ignore
        ax.barh(
            pair_labels[::-1],
            pair_counts[::-1],
            color=colors[::-1],
            edgecolor="white",
            alpha=0.8,
        )
        for i, c in enumerate(pair_counts[::-1]):
            ax.text(c + max(pair_counts) * 0.01, i, str(c), va="center", fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title("Speaker Transitions (who→who)")

    # 3. Turns per conversation histogram (multi-speaker only)
    ax = axes[0, 2]
    if len(conv_df) > 0:
        turns_per_conv = conv_df["n_turns"].to_numpy()
        max_t = min(int(np.percentile(turns_per_conv, 99)), 50) + 1
        ax.hist(
            turns_per_conv,
            bins=range(1, max_t + 1),
            color="#55A868",
            edgecolor="white",
            alpha=0.8,
            align="left",
        )
        ax.axvline(
            np.median(turns_per_conv),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(turns_per_conv):.0f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Turns per conversation")
    ax.set_ylabel("Count")
    ax.set_title("Turns per Conversation\n(multi-speaker only)")

    # 4. Response latency by transition type (box)
    ax = axes[1, 0]
    if len(transition_df) > 0:
        # Top transition pairs
        top_pairs = (
            transition_df.group_by(["from_label", "to_label"])
            .len()
            .sort("len", descending=True)
            .head(6)
        )
        box_data = []
        box_labels: list[str] = []
        for row in top_pairs.iter_rows(named=True):
            fl, tl = row["from_label"], row["to_label"]
            gaps = transition_df.filter(
                (pl.col("from_label") == fl) & (pl.col("to_label") == tl)
            )["gap_s"].to_numpy()
            # Clip for visibility
            gaps = np.clip(gaps, -1, 10)
            box_data.append(gaps)
            box_labels.append(f"{fl}→{tl}")
        if box_data:
            ax.boxplot(
                box_data,
                labels=box_labels,
                patch_artist=True,
                widths=0.5,
                boxprops=dict(alpha=0.7),
                medianprops=dict(color="red"),
            )
            ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Response latency (s)")
    ax.set_title("Response Latency by Transition")

    # 5. Conversations vs monologues — count and total duration
    ax = axes[1, 1]
    n_conv = len(conv_df)
    n_mono = len(mono_df)
    h_conv = float(conv_df["duration"].sum()) / 3600 if n_conv > 0 else 0.0
    h_mono = float(mono_df["duration"].sum()) / 3600 if n_mono > 0 else 0.0
    x = np.array([0, 1])
    bars = ax.bar(
        x,
        [n_conv, n_mono],
        color=["#4C72B0", "#CCB974"],
        width=0.5,
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Conversations\n({h_conv:.1f}h)", f"Monologues\n({h_mono:.1f}h)"]
    )
    ax.set_ylabel("Count")
    for bar, val in zip(bars, [n_conv, n_mono]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(n_conv, n_mono) * 0.01,
            str(val),
            ha="center",
            fontsize=9,
        )
    ax.set_title("Conversations vs Monologues")

    # 6. Conversation duration histogram (multi-speaker only)
    ax = axes[1, 2]
    if len(conv_df) > 0:
        cd = conv_df["duration"].to_numpy()
        p99_cd = float(np.percentile(cd, 99))
        n_clipped_cd = int((cd > p99_cd).sum())
        if n_clipped_cd:
            print(f"  [conv duration multi-speaker] {n_clipped_cd} convs > p99={p99_cd:.1f}s clipped")
        ax.hist(np.clip(cd, 0, p99_cd), bins=50, color="#DA8BC3", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(cd),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(cd):.1f}s",
        )
        ax.legend(fontsize=8)
        if n_clipped_cd:
            ax.text(0.98, 0.97, f"{n_clipped_cd} > p99={p99_cd:.0f}s",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.set_xlabel(f"Conversation duration (s, p99={p99_cd:.0f}s)" if len(conv_df) > 0 else "Conversation duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Conversation Duration\n(multi-speaker only)")

    # 7. Speech overlap — simultaneous speech between speaker pairs
    ax = axes[2, 0]
    overlap_min = _compute_label_overlap_minutes(segment_df)
    if overlap_min:
        pairs = sorted(overlap_min, key=lambda k: overlap_min[k], reverse=True)
        vals = [overlap_min[p] for p in pairs]
        bars = ax.barh(
            pairs[::-1], vals[::-1], color="#C44E52", edgecolor="white", alpha=0.8
        )
        x_max = max(vals) if vals else 1
        for bar, v in zip(bars, vals[::-1]):
            ax.text(
                v + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} min",
                va="center",
                fontsize=8,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No simultaneous speech detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Total simultaneous speech (minutes)")
    ax.set_title("Speech Overlap by Speaker Pair")

    # 8. Silence structure — inter-conversation gap distribution (multi-speaker)
    ax = axes[2, 1]
    if len(conv_df) > 0:
        ic_gaps = conv_df["gap_after"].drop_nulls().to_numpy()
        if len(ic_gaps) > 0:
            ic_clipped = np.clip(ic_gaps, 0, np.percentile(ic_gaps, 99))
            ax.hist(ic_clipped, bins=50, color="#8172B2", edgecolor="white", alpha=0.8)
            ax.axvline(
                np.median(ic_gaps),
                color="red",
                ls="--",
                lw=1,
                label=f"median={np.median(ic_gaps):.1f}s",
            )
            ax.legend(fontsize=8)
    ax.set_xlabel("Inter-conversation silence (s)")
    ax.set_ylabel("Count")
    ax.set_title("Silence Between Conversations\n(multi-speaker only)")

    # 9. Monologue analysis — duration distribution + dominant label breakdown
    ax = axes[2, 2]
    if len(mono_df) > 0 and len(conv_df) > 0:
        # Overlaid duration histograms: conversations vs monologues
        c_dur = conv_df["duration"].to_numpy()
        m_dur = mono_df["duration"].to_numpy()
        p99 = np.percentile(np.concatenate([c_dur, m_dur]), 99)
        ax.hist(
            np.clip(c_dur, 0, p99),
            bins=40,
            alpha=0.5,
            color="#4C72B0",
            label=f"Conv (med={np.median(c_dur):.1f}s)",
            weights=np.ones_like(c_dur) * 100 / len(c_dur),
        )
        ax.hist(
            np.clip(m_dur, 0, p99),
            bins=40,
            alpha=0.5,
            color="#CCB974",
            label=f"Mono (med={np.median(m_dur):.1f}s)",
            weights=np.ones_like(m_dur) * 100 / len(m_dur),
        )
        ax.legend(fontsize=8)
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("% of events")
    elif len(mono_df) > 0:
        m_dur = mono_df["duration"].to_numpy()
        ax.hist(
            np.clip(m_dur, 0, np.percentile(m_dur, 99)),
            bins=40,
            color="#CCB974",
            edgecolor="white",
            alpha=0.8,
        )
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
    # Dominant label counts as text annotation
    if len(mono_df) > 0 and "labels" in mono_df.columns:
        label_counts = mono_df.group_by("labels").len().sort("len", descending=True)
        lines = [
            f"{r['labels']}: {r['len']}" for r in label_counts.iter_rows(named=True)
        ]
        ax.text(
            0.97,
            0.97,
            "By label\n" + "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
    ax.set_title("Monologue vs Conversation Duration")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")


# =========================================================================
# Page 3: Turns & Conversations (3×3)
#   Turn duration per role, conversation duration, turns/conv,
#   inter-conversation duration, conversation SNR/C50
# =========================================================================


def save_boss_figures(
    turn_df: pl.DataFrame,
    conversation_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Boss-requested plots: turn/conversation histograms + SNR/C50.

    Turn definition: VTC activity surrounded by >300ms non-activity.
    Conversation definition: turns with <10s silence between them.
    """
    plt = _setup()

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        "Turns & Conversations\n"
        "(turn = VTC activity with >300 ms silence boundary;  "
        "conversation = turns with <10 s gap)",
        fontsize=13,
        fontweight="bold",
    )

    # ── Row 1: Turn duration per VTC role ──────────────────────────

    # 1. Turn duration — all roles overlaid
    ax = axes[0, 0]
    if len(turn_df) > 0:
        all_d = turn_df["duration"].to_numpy()
        p99 = float(np.percentile(all_d, 99))
        n_clipped_total = int((all_d > p99).sum())
        for l in VTC_LABELS:
            d = turn_df.filter(pl.col("label") == l)["duration"].to_numpy()
            if len(d) > 0:
                ax.hist(
                    np.clip(d, 0, p99),
                    bins=60,
                    alpha=0.5,
                    color=LABEL_COLORS.get(l, "#999"),
                    label=f"{l} (n={len(d)}, med={np.median(d):.1f}s)",
                    edgecolor="white",
                )
        if n_clipped_total:
            print(
                f"  [turn duration] {n_clipped_total} turns > p99={p99:.1f}s clipped from overlaid plot"
            )
        ax.legend(fontsize=7)
    ax.set_xlabel(
        f"Turn duration (s, capped at p99={p99:.0f}s)"
        if len(turn_df) > 0
        else "Turn duration (s)"
    )
    ax.set_ylabel("Count")
    ax.set_title("Turn Duration by Role (overlaid)")

    # 2–5: Individual per-role histograms (2×2 in positions [0,1], [0,2], [1,0], [1,1])
    role_positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    for (row_i, col_i), l in zip(role_positions, VTC_LABELS):
        ax = axes[row_i, col_i]
        d = (
            turn_df.filter(pl.col("label") == l)["duration"].to_numpy()
            if len(turn_df) > 0
            else np.array([])
        )
        if len(d) > 0:
            p99_l = float(np.percentile(d, 99))
            n_clipped = int((d > p99_l).sum())
            if n_clipped:
                print(
                    f"  [turn duration {l}] {n_clipped} turns > p99={p99_l:.1f}s clipped"
                )
            ax.hist(
                np.clip(d, 0, p99_l),
                bins=60,
                color=LABEL_COLORS.get(l, "#999"),
                edgecolor="white",
                alpha=0.8,
            )
            ax.axvline(
                np.median(d),
                color="red",
                ls="--",
                lw=1,
                label=f"median={np.median(d):.1f}s",
            )
            ax.legend(fontsize=8)
            ax.text(
                0.98,
                0.95,
                f"n={len(d)}\nmean={np.mean(d):.1f}s\nstd={np.std(d):.1f}s"
                + (f"\n({n_clipped} > p99={p99_l:.0f}s)" if n_clipped else ""),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7),
            )
        ax.set_xlabel(
            f"Turn duration (s, p99={p99_l:.0f}s)"
            if len(d) > 0
            else "Turn duration (s)"
        )
        ax.set_ylabel("Count")
        ax.set_title(f"Turn Duration — {l}")

    # 6. Conversation duration histogram
    ax = axes[1, 2]
    if len(conversation_df) > 0:
        cd = conversation_df["duration"].to_numpy()
        p99_cd = float(np.percentile(cd, 99))
        n_clipped_cd = int((cd > p99_cd).sum())
        if n_clipped_cd:
            print(f"  [conv duration] {n_clipped_cd} convs > p99={p99_cd:.1f}s clipped")
        ax.hist(np.clip(cd, 0, p99_cd), bins=50, color="#DA8BC3", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(cd),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(cd):.1f}s",
        )
        ax.legend(fontsize=8)
        ax.text(
            0.98,
            0.95,
            f"n={len(cd)}\nmean={np.mean(cd):.1f}s\nstd={np.std(cd):.1f}s"
            + (f"\n({n_clipped_cd} > p99={p99_cd:.0f}s)" if n_clipped_cd else ""),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )
    ax.set_xlabel(f"Conversation duration (s, p99={p99_cd:.0f}s)" if len(conversation_df) > 0 else "Conversation duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Conversation Duration")

    # 7. Number of turns per conversation
    ax = axes[2, 0]
    if len(conversation_df) > 0:
        nt = conversation_df["n_turns"].to_numpy()
        max_bin = min(int(np.percentile(nt, 99)), 40) + 1
        ax.hist(
            nt,
            bins=range(1, max_bin + 1),
            color="#64B5CD",
            edgecolor="white",
            alpha=0.8,
            align="left",
        )
        ax.axvline(
            np.median(nt),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(nt):.0f}",
        )
        ax.legend(fontsize=8)
        ax.text(
            0.98,
            0.95,
            f"n={len(nt)}\nmean={np.mean(nt):.1f}\nstd={np.std(nt):.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )
    ax.set_xlabel("Turns per conversation")
    ax.set_ylabel("Count")
    ax.set_title("Turns per Conversation")

    # 8. Inter-conversation duration
    ax = axes[2, 1]
    if len(conversation_df) > 0:
        ic_gaps = conversation_df["gap_after"].drop_nulls().to_numpy()
        if len(ic_gaps) > 0:
            ic_clip = np.clip(ic_gaps, 0, min(float(np.percentile(ic_gaps, 99)), 600))
            ax.hist(ic_clip, bins=50, color="#CCB974", edgecolor="white", alpha=0.8)
            ax.axvline(
                np.median(ic_gaps),
                color="red",
                ls="--",
                lw=1,
                label=f"median={np.median(ic_gaps):.1f}s",
            )
            ax.legend(fontsize=8)
            ax.text(
                0.98,
                0.95,
                f"n={len(ic_gaps)}\nmean={np.mean(ic_gaps):.1f}s\n"
                f"std={np.std(ic_gaps):.1f}s",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7),
            )
    ax.set_xlabel("Inter-conversation silence (s)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-conversation Duration")

    # 9. Conversation SNR + C50 (dual histogram)
    ax = axes[2, 2]
    if "snr_mean" in conversation_df.columns:
        conv_snr = conversation_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
        if len(conv_snr) > 0:
            ax.hist(
                conv_snr,
                bins=40,
                color="#4C72B0",
                edgecolor="white",
                alpha=0.6,
                label=f"SNR (med={np.median(conv_snr):.1f})",
            )
    if "c50_mean" in conversation_df.columns:
        conv_c50 = conversation_df["c50_mean"].drop_nulls().drop_nans().to_numpy()
        if len(conv_c50) > 0:
            ax.hist(
                conv_c50,
                bins=40,
                color="#55A868",
                edgecolor="white",
                alpha=0.6,
                label=f"C50 (med={np.median(conv_c50):.1f})",
            )
    ax.set_xlabel("dB")
    ax.set_ylabel("Count")
    ax.set_title("Conversation SNR & C50")
    ax.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")
