"""Master dashboard figures — two 3×3 summary PNGs.

Master 1  «Dataset & Content Overview»
  Row 0  Content:     Speech volume by label, clip duration, cut-tier breakdown
  Row 1  Clip quality: VAD–VTC IoU, speech density, speaker turns per clip
  Row 2  Structure:   VTC segment duration, label diversity, dataset summary card

Master 2  «Recording & Conversation Quality»
  Row 0  Turns:       Turn duration by role, speaker transitions, turn density
  Row 1  Conversations: conversations vs monologues, conversation duration,
                        turns per conversation
  Row 2  Recording:   Per-segment SNR & C50 (overlaid), ESC active rate bars,
                       ESC category bars (pw-hours)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.core import LABEL_COLORS, VTC_LABELS
from src.plotting.utils import lazy_pyplot, save_figure


# ═══════════════════════════════════════════════════════════════════════════
# ESC colour map (shared with snr_noise.py)
# ═══════════════════════════════════════════════════════════════════════════

_NOISE_COLORS: dict[str, str] = {
    "music": "#E24A33",
    "crying": "#348ABD",
    "laughter": "#FBC15E",
    "singing": "#8EBA42",
    "tv_radio": "#988ED5",
    "vehicle": "#777777",
    "animal": "#FF8C42",
    "household": "#C44E52",
    "impact": "#8C8C8C",
    "silence": "#64B5CD",
    "human_activity": "#FFB347",
    "nature": "#55A868",
    "machinery": "#A0826D",
    "alarm_signal": "#CCB974",
    "environment": "#5B7FA6",
    "other": "#D3D3D3",
}

_SNR_COLOR = "#4C72B0"
_C50_COLOR = "#DD8452"


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _summary_card(ax, clip_df: pl.DataFrame, file_df: pl.DataFrame) -> None:
    """Render a dataset summary info card on a hidden axis."""
    from matplotlib.patches import FancyBboxPatch

    ax.axis("off")

    n_clips = len(clip_df)
    total_dur_h = clip_df["duration"].sum() / 3600
    total_vtc_h = clip_df["vtc_speech_dur"].sum() / 3600
    total_vad_h = clip_df["vad_speech_dur"].sum() / 3600
    n_files = len(file_df) if len(file_df) > 0 else clip_df["uid"].n_unique()
    median_dur = float(np.median(clip_df["duration"].to_numpy()))
    median_iou = float(np.median(clip_df["vad_vtc_iou"].to_numpy()))
    n_convs = int(clip_df["n_conversations"].sum())
    n_turns = int(clip_df["n_conv_turns"].sum())

    rows: list[tuple[bool, str, str]] = [
        (True, "FILES & CLIPS", ""),
        (False, "Source files", f"{n_files:,}"),
        (False, "Total clips", f"{n_clips:,}"),
        (True, "AUDIO", ""),
        (False, "Total duration", f"{total_dur_h:.1f} h"),
        (False, "VTC speech", f"{total_vtc_h:.1f} h"),
        (False, "VAD speech", f"{total_vad_h:.1f} h"),
        (True, "CLIP QUALITY", ""),
        (False, "Median duration", f"{median_dur:.0f} s"),
        (False, "Median IoU", f"{median_iou:.2f}"),
        (True, "CONVERSATIONS", ""),
        (False, "Total turns", f"{n_turns:,}"),
        (False, "Total conversations", f"{n_convs:,}"),
    ]

    snr_vals = clip_df["snr_mean"].drop_nulls().drop_nans()
    c50_vals = clip_df["c50_mean"].drop_nulls().drop_nans()
    if len(snr_vals) > 0 or len(c50_vals) > 0:
        rows.append((True, "RECORDING", ""))
        if len(snr_vals) > 0:
            rows.append(
                (False, "Median SNR", f"{float(snr_vals.median()):.1f} dB")  # type: ignore
            )
        if len(c50_vals) > 0:
            rows.append((False, "Median C50", f"{float(c50_vals.median()):.1f} dB"))  # type: ignore

    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.04),
            0.92,
            0.92,
            boxstyle="round,pad=0.02",
            transform=ax.transAxes,
            facecolor="#f8f9fa",
            edgecolor="#ced4da",
            lw=1.5,
            zorder=0,
        )
    )
    ax.set_title("Dataset Summary")

    y_top, y_bot = 0.96, 0.08
    step = (y_top - y_bot) / len(rows)
    for i, (is_hdr, label, value) in enumerate(rows):
        y = y_top - (i + 0.5) * step
        if is_hdr:
            ax.text(
                0.08,
                y,
                label,
                transform=ax.transAxes,
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#868e96",
            )
        else:
            ax.text(
                0.12,
                y,
                label,
                transform=ax.transAxes,
                va="center",
                fontsize=9,
                color="#495057",
            )
            ax.text(
                0.92,
                y,
                value,
                transform=ax.transAxes,
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
                color="#212529",
            )


def _hist_pair(ax, snr_vals: np.ndarray, c50_vals: np.ndarray) -> None:
    """Overlay SNR and C50 histograms on *ax*, with mean annotations."""
    total_n = len(snr_vals) + len(c50_vals)
    bins = max(8, int(np.ceil(np.log2(total_n) + 1))) if total_n > 0 else 8
    if len(snr_vals) > 0:
        ax.hist(
            snr_vals,
            bins=bins,
            color=_SNR_COLOR,
            alpha=0.65,
            edgecolor="white",
            label=f"SNR  mean={np.mean(snr_vals):.1f} dB",
        )
    if len(c50_vals) > 0:
        ax.hist(
            c50_vals,
            bins=bins,
            color=_C50_COLOR,
            alpha=0.65,
            edgecolor="white",
            label=f"C50  mean={np.mean(c50_vals):.1f} dB",
        )
    ax.set_xlabel("dB")
    ax.set_ylabel("Count")
    if len(snr_vals) > 0 or len(c50_vals) > 0:
        ax.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
# Master 1 — Dataset & Content Overview
# ═══════════════════════════════════════════════════════════════════════════


def save_master_overview(
    clip_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    file_df: pl.DataFrame,
    tier_counts: dict[str, int],
    output_path: Path,
) -> None:
    """Master overview dashboard — 3×3 panels.

    Row 0  Content:      Speech volume by label, clip duration, cut-tier breakdown
    Row 1  Clip quality:  VAD–VTC IoU, speech density, speaker turns per clip
    Row 2  Structure:     VTC segment duration, label diversity, dataset summary
    """
    plt = lazy_pyplot()

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Dataset & Content Overview", fontsize=14, fontweight="bold")

    # ── Row 0: Content ──────────────────────────────────────────────

    # [0,0] Speech volume by label
    ax = axes[0, 0]
    hours = []
    for l in VTC_LABELS:
        col = f"dur_{l}"
        hours.append(clip_df[col].sum() / 3600 if col in clip_df.columns else 0)
    bars = ax.bar(
        VTC_LABELS,
        hours,
        color=[LABEL_COLORS[l] for l in VTC_LABELS],
        edgecolor="white",
    )
    for bar, h in zip(bars, hours):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}h",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Total speech (hours)")
    ax.set_title("Speech Volume by Speaker Type")

    # [0,1] Clip duration distribution
    ax = axes[0, 1]
    durs = clip_df["duration"].to_numpy()
    ax.hist(durs, bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.median(durs),
        color="red",
        ls="--",
        lw=1,
        label=f"median={np.median(durs):.0f}s",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("Clip duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Clip Duration Distribution")

    # [0,2] Cut-tier breakdown
    ax = axes[0, 2]
    _tier_labels = {
        "long_union_gap": "1. Long silence",
        "short_union_gap": "2. Short silence",
        "vad_only_gap": "3. VAD-only gap",
        "vtc_only_gap": "4. VTC-only gap",
        "speaker_boundary": "5. Speaker boundary",
        "hard_cut": "6. Hard cut",
        "degenerate_window": "7. Degenerate",
    }
    _tier_hex = {
        "long_union_gap": "#2ecc71",
        "short_union_gap": "#27ae60",
        "vad_only_gap": "#f1c40f",
        "vtc_only_gap": "#f39c12",
        "speaker_boundary": "#e67e22",
        "hard_cut": "#e74c3c",
        "degenerate_window": "#c0392b",
    }
    total_cuts = sum(tier_counts.values())
    if total_cuts > 0:
        names, vals, colors = [], [], []
        for k in _tier_labels:
            cnt = tier_counts.get(k, 0)
            if cnt > 0:
                names.append(_tier_labels[k])
                vals.append(cnt)
                colors.append(_tier_hex.get(k, "#999"))
        ax.barh(
            names[::-1],
            vals[::-1],
            color=colors[::-1],
            edgecolor="white",
            alpha=0.8,
        )
        for i, v in enumerate(vals[::-1]):
            pct = 100 * v / total_cuts
            ax.text(
                v + total_cuts * 0.01,
                i,
                f"{v} ({pct:.1f}%)",
                va="center",
                fontsize=8,
            )
    ax.set_xlabel("Number of cuts")
    ax.set_title(f"Cut-Point Tier Breakdown ({total_cuts} total)")

    # ── Row 1: Clip quality ─────────────────────────────────────────

    # [1,0] VAD–VTC IoU distribution
    ax = axes[1, 0]
    ious = clip_df["vad_vtc_iou"].to_numpy()
    n_zero = int((ious == 0).sum())
    ax.hist(ious, bins=40, color="#8172B2", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.median(ious),
        color="red",
        ls="--",
        lw=1,
        label=f"median={np.median(ious):.2f}",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("VAD–VTC IoU (per clip)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"VAD–VTC Agreement (IoU)\n"
        f"{n_zero} clips with IoU=0 (near-silent)",
        fontsize=10,
    )

    # [1,1] Per-clip VTC speech density histogram
    ax = axes[1, 1]
    vtc_dens = clip_df["vtc_density"].to_numpy()
    ax.hist(vtc_dens, bins=40, color="#55A868", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.median(vtc_dens),
        color="red",
        ls="--",
        lw=1,
        label=f"median={np.median(vtc_dens):.2f}",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("VTC speech density")
    ax.set_ylabel("Count")
    ax.set_title("Speech Density per Clip")

    # [1,2] Turns per clip histogram
    ax = axes[1, 2]
    turns = clip_df["n_conv_turns"].to_numpy()
    ax.hist(
        turns,
        bins=50,
        color="#CCB974",
        edgecolor="white",
        alpha=0.8,
    )
    ax.axvline(
        np.median(turns),
        color="red",
        ls="--",
        lw=1,
        label=f"median={np.median(turns):.0f}",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("Speaker turns")
    ax.set_ylabel("Count")
    ax.set_title("Speaker Turns per Clip")

    # ── Row 2: Structure ────────────────────────────────────────────

    # [2,0] VTC segment duration histogram
    ax = axes[2, 0]
    seg_durs = segment_df["duration"].to_numpy()
    clipped = np.clip(seg_durs, 0, 30)
    ax.hist(clipped, bins=60, color="#DD8452", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.median(seg_durs),
        color="red",
        ls="--",
        lw=1,
        label=f"median={np.median(seg_durs):.1f}s",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("VTC segment duration (s, capped at 30)")
    ax.set_ylabel("Count")
    ax.set_title("VTC Segment Duration")

    # [2,1] Label diversity per clip
    ax = axes[2, 1]
    n_labels = clip_df["n_labels"].to_numpy()
    ax.hist(
        n_labels,
        bins=range(0, 6),
        color="#64B5CD",
        edgecolor="white",
        alpha=0.8,
        align="left",
    )
    ax.set_xlabel("Unique labels")
    ax.set_ylabel("Count")
    ax.set_title("Label Diversity per Clip")
    ax.set_xticks(range(0, 5))

    # [2,2] Dataset summary card
    _summary_card(axes[2, 2], clip_df, file_df)

    plt.tight_layout()
    save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Master 2 — Recording & Conversation Quality
# ═══════════════════════════════════════════════════════════════════════════


def _esc_horiz_bars(ax, categories: list[str], values: np.ndarray,
                      title: str, xlabel: str, *, pct: bool = False) -> None:
    """Horizontal bar chart for ESC categories, coloured per category."""
    order = np.argsort(values)
    sorted_labels = [categories[i] for i in order]
    sorted_values = values[order]
    colors = [_NOISE_COLORS.get(l, "#D3D3D3") for l in sorted_labels]

    ax.barh(sorted_labels, sorted_values, color=colors, edgecolor="white", height=0.7)
    for bar, val in zip(ax.patches, sorted_values):
        label_str = (
            f"{val:.1f}%" if pct else f"{val:.3f}" if val < 0.1 else f"{val:.2f}"
        )
        ax.text(
            bar.get_width() + max(sorted_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label_str,
            va="center",
            fontsize=7,
        )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(0, max(sorted_values) * 1.20 if sorted_values.max() > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_master_quality(
    clip_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    turn_df: pl.DataFrame,
    conversation_df: pl.DataFrame,
    transition_df: pl.DataFrame,
    output_path: Path,
    esc_stats_dir: Path | None = None,
) -> None:
    """Recording & conversation quality dashboard — 3×3 panels.

    Row 0  Turns:         Turn duration by role, speaker transitions, turn density
    Row 1  Conversations: convs vs monologues, conversation duration, turns/conv
    Row 2  Recording:     Per-segment SNR/C50, ESC active rate, ESC pw-hours
    """
    plt = lazy_pyplot()

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        "Recording & Conversation Quality",
        fontsize=14,
        fontweight="bold",
    )

    # ── Row 0: Turns ────────────────────────────────────────────────

    # [0,0] Turn duration by role (overlaid histogram)
    ax = axes[0, 0]
    if len(turn_df) > 0:
        all_d = turn_df["duration"].to_numpy()
        p99 = float(np.percentile(all_d, 99))
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
        ax.legend(fontsize=7)
        ax.set_xlabel(f"Turn duration (s, capped at p99={p99:.0f}s)")
    else:
        ax.set_xlabel("Turn duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Turn Duration by Role")

    # [0,1] Speaker transitions (who→who)
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
        max_c = max(pair_counts)
        for i, c in enumerate(pair_counts[::-1]):
            ax.text(
                c + max_c * 0.01,
                i,
                f"{c:,}",
                va="center",
                fontsize=7,
            )
        ax.set_xlim(0, max_c * 1.18)
        ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Count")
    ax.set_title("Speaker Transitions (who→who)")

    # [0,2] Turn density per clip (turns per minute)
    ax = axes[0, 2]
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

    # ── Row 1: Conversations ────────────────────────────────────────

    conv_df = conversation_df.filter(pl.col("is_multi_speaker") == True)  # noqa: E712
    mono_df = conversation_df.filter(pl.col("is_multi_speaker") == False)  # noqa: E712

    # [1,0] Conversations vs monologues
    ax = axes[1, 0]
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
        [f"Conversations\n({h_conv:.1f}h)", f"Monologues\n({h_mono:.1f}h)"],
    )
    ax.set_ylabel("Count")
    for bar, val in zip(bars, [n_conv, n_mono]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(n_conv, n_mono, 1) * 0.01,
            str(val),
            ha="center",
            fontsize=9,
        )
    ax.set_title("Conversations vs Monologues")

    # [1,1] Conversation duration histogram
    ax = axes[1, 1]
    if len(conversation_df) > 0:
        cd = conversation_df["duration"].to_numpy()
        p99_cd = float(np.percentile(cd, 99))
        ax.hist(
            np.clip(cd, 0, p99_cd),
            bins=50,
            color="#DA8BC3",
            edgecolor="white",
            alpha=0.8,
        )
        ax.axvline(
            np.median(cd),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(cd):.1f}s",
        )
        ax.legend(fontsize=8)
        ax.set_xlabel(f"Conversation duration (s, p99={p99_cd:.0f}s)")
    else:
        ax.set_xlabel("Conversation duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Conversation Duration")

    # [1,2] Turns per conversation
    ax = axes[1, 2]
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
    ax.set_xlabel("Turns per conversation")
    ax.set_ylabel("Count")
    ax.set_title("Turns per Conversation")

    # ── Row 2: Recording quality ────────────────────────────────────

    # [2,0] Per-segment SNR & C50 overlapping histograms
    ax = axes[2, 0]
    seg_snr = segment_df["snr_during"].drop_nulls().to_numpy()
    seg_c50 = segment_df["c50_during"].drop_nulls().to_numpy()
    _hist_pair(ax, seg_snr, seg_c50)
    ax.set_title("Per-Segment SNR & C50\n(VTC speech frames only)", fontsize=10)

    # [2,1] Noise categories — horizontal bars (active detection rate)
    ax = axes[2, 1]
    cat_stats_path = (
        esc_stats_dir / "category_stats.parquet"
        if esc_stats_dir is not None
        else None
    )
    _esc_cat_df = None
    if cat_stats_path is not None and cat_stats_path.exists():
        _esc_cat_df = (
            pl.read_parquet(cat_stats_path)
            .filter(pl.col("map_version") == "new")
            .sort("pw_hours", descending=True)
        )
        cats = _esc_cat_df["category"].to_list()
        active = _esc_cat_df["active_rate"].to_numpy().astype(np.float64) * 100
        _esc_horiz_bars(
            ax, cats, active,
            "Noise Active Rate\n(% of 1-s windows with P > 0.05)", "% of windows",
            pct=True,
        )
    else:
        ax.text(
            0.5, 0.5, "No ESC data", ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="#868e96",
        )
        ax.set_title("Noise Active Rate", fontsize=10)
        ax.axis("off")

    # [2,2] Noise categories — horizontal bars (pw-hours)
    ax = axes[2, 2]
    if _esc_cat_df is not None:
        cats = _esc_cat_df["category"].to_list()
        hrs = _esc_cat_df["pw_hours"].to_numpy().astype(np.float64)
        _esc_horiz_bars(
            ax, cats, hrs, "Noise Categories\n(probability-weighted hours)", "Hours"
        )
    else:
        ax.text(
            0.5, 0.5, "No ESC data", ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="#868e96",
        )
        ax.set_title("Noise Categories", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Text summary — printed to stdout for SLURM log parsing
# ═══════════════════════════════════════════════════════════════════════════


def print_dataset_summary(
    dfs: dict[str, pl.DataFrame],
    tier_counts: dict[str, int],
) -> None:
    """Print key dataset/plot metrics to stdout.

    This makes SLURM logs interpretable without inspecting images.
    """
    clip_df = dfs["clip_stats"]
    segment_df = dfs["segment_stats"]
    turn_df = dfs["turn_stats"]
    conv_df = dfs["conversation_stats"]
    trans_df = dfs["transition_stats"]
    file_df = dfs["file_stats"]

    sep = "━" * 60

    # ── Dataset overview ──────────────────────────────────────────
    print(f"\n{sep}")
    print("DATASET OVERVIEW")
    print(sep)
    n_clips = len(clip_df)
    n_files = clip_df["uid"].n_unique()
    total_dur_h = clip_df["duration"].sum() / 3600
    total_vtc_h = clip_df["vtc_speech_dur"].sum() / 3600
    total_vad_h = clip_df["vad_speech_dur"].sum() / 3600
    print(f"  Files            : {n_files:,}")
    print(f"  Clips            : {n_clips:,}")
    print(f"  Total audio      : {total_dur_h:.2f} h")
    print(f"  VTC speech       : {total_vtc_h:.2f} h")
    print(f"  VAD speech       : {total_vad_h:.2f} h")
    durs = clip_df["duration"].to_numpy()
    print(f"  Clip duration    : median={np.median(durs):.0f}s  "
          f"mean={np.mean(durs):.0f}s  std={np.std(durs):.0f}s  "
          f"min={np.min(durs):.0f}s  max={np.max(durs):.0f}s")
    iou = clip_df["vad_vtc_iou"].to_numpy()
    print(f"  VAD-VTC IoU      : median={np.median(iou):.3f}  mean={np.mean(iou):.3f}")

    # ── Per-label speech hours ────────────────────────────────────
    print(f"\n{sep}")
    print("SPEECH VOLUME BY LABEL")
    print(sep)
    for l in VTC_LABELS:
        col = f"dur_{l}"
        if col in clip_df.columns:
            h = clip_df[col].sum() / 3600
            n_seg = int(segment_df.filter(pl.col("label") == l).height) if len(segment_df) > 0 else 0
            print(f"  {l:6s} : {h:7.2f} h   ({n_seg:,} segments)")
    cf = clip_df["child_fraction"].to_numpy()
    print(f"  Child fraction   : median={np.median(cf):.3f}  mean={np.mean(cf):.3f}")

    # ── SNR & C50 ─────────────────────────────────────────────────
    snr_vals = clip_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
    c50_vals = clip_df["c50_mean"].drop_nulls().drop_nans().to_numpy()
    if len(snr_vals) > 0 or len(c50_vals) > 0:
        print(f"\n{sep}")
        print("SNR & C50 (CLIP-LEVEL)")
        print(sep)
    if len(snr_vals) > 0:
        print(f"  SNR  (n={len(snr_vals):,}) : median={np.median(snr_vals):.1f} dB  "
              f"mean={np.mean(snr_vals):.1f}  std={np.std(snr_vals):.1f}  "
              f"[{np.min(snr_vals):.1f}, {np.max(snr_vals):.1f}]")
        for t in [0, 5, 10, 15, 20]:
            frac = float((snr_vals < t).mean())
            print(f"    SNR < {t:2d} dB : {frac:.1%} of clips")
        if "snr_during" in segment_df.columns:
            seg_snr = segment_df.filter(pl.col("snr_during").is_not_null())
            if len(seg_snr) > 0:
                print("  Per-label SNR during speech:")
                for l in VTC_LABELS:
                    v = seg_snr.filter(pl.col("label") == l)["snr_during"].to_numpy()
                    if len(v) > 0:
                        print(f"    {l:6s} : mean={np.mean(v):.1f} dB  std={np.std(v):.1f}")
    if len(c50_vals) > 0:
        print(f"  C50  (n={len(c50_vals):,}) : median={np.median(c50_vals):.1f} dB  "
              f"mean={np.mean(c50_vals):.1f}  std={np.std(c50_vals):.1f}  "
              f"[{np.min(c50_vals):.1f}, {np.max(c50_vals):.1f}]")

    # ── Turns ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("TURNS (gap merge = 300ms)")
    print(sep)
    n_turns = len(turn_df)
    print(f"  Total turns      : {n_turns:,}")
    if n_turns > 0:
        td = turn_df["duration"].to_numpy()
        print(f"  Turn duration    : median={np.median(td):.2f}s  "
              f"mean={np.mean(td):.2f}s  std={np.std(td):.2f}s")
        for l in VTC_LABELS:
            d = turn_df.filter(pl.col("label") == l)["duration"].to_numpy()
            if len(d) > 0:
                print(f"    {l:6s} (n={len(d):,}) : "
                      f"median={np.median(d):.2f}s  mean={np.mean(d):.2f}s  "
                      f"std={np.std(d):.2f}s")
        td_per_min = clip_df["turn_density_per_min"].drop_nulls().to_numpy()
        if len(td_per_min) > 0:
            print(f"  Turn density     : median={np.median(td_per_min):.1f}/min  "
                  f"mean={np.mean(td_per_min):.1f}/min")

    # ── Conversations ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("CONVERSATIONS (max silence = 10s)")
    print(sep)
    n_convs = len(conv_df)
    print(f"  Total conversations : {n_convs:,}")
    if n_convs > 0:
        cd = conv_df["duration"].to_numpy()
        print(f"  Duration         : median={np.median(cd):.1f}s  "
              f"mean={np.mean(cd):.1f}s  std={np.std(cd):.1f}s  "
              f"[{np.min(cd):.1f}, {np.max(cd):.1f}]")
        nt = conv_df["n_turns"].to_numpy()
        print(f"  Turns/conv       : median={np.median(nt):.0f}  "
              f"mean={np.mean(nt):.1f}  std={np.std(nt):.1f}  "
              f"max={np.max(nt)}")
        multi = int(conv_df["is_multi_speaker"].sum())
        print(f"  Multi-speaker    : {multi:,} ({100*multi/n_convs:.1f}%)")
        print(f"  Single-speaker   : {n_convs - multi:,} ({100*(n_convs-multi)/n_convs:.1f}%)")
        ic_gaps = conv_df["gap_after"].drop_nulls().to_numpy()
        if len(ic_gaps) > 0:
            print(f"  Inter-conv gap   : median={np.median(ic_gaps):.1f}s  "
                  f"mean={np.mean(ic_gaps):.1f}s  std={np.std(ic_gaps):.1f}s")
        if "snr_mean" in conv_df.columns:
            cs = conv_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
            if len(cs) > 0:
                print(f"  Conv SNR         : median={np.median(cs):.1f} dB  "
                      f"mean={np.mean(cs):.1f}")
        if "c50_mean" in conv_df.columns:
            cc = conv_df["c50_mean"].drop_nulls().drop_nans().to_numpy()
            if len(cc) > 0:
                print(f"  Conv C50         : median={np.median(cc):.1f} dB  "
                      f"mean={np.mean(cc):.1f}")

    # ── Speaker transitions ───────────────────────────────────────
    print(f"\n{sep}")
    print("SPEAKER TRANSITIONS")
    print(sep)
    n_trans = len(trans_df)
    print(f"  Total transitions: {n_trans:,}")
    if n_trans > 0:
        top = (
            trans_df.group_by(["from_label", "to_label"])
            .len()
            .sort("len", descending=True)
            .head(10)
        )
        for row in top.iter_rows(named=True):
            fl, tl = row["from_label"], row["to_label"]
            cnt = row["len"]
            gaps = trans_df.filter(
                (pl.col("from_label") == fl) & (pl.col("to_label") == tl)
            )["gap_s"].to_numpy()
            print(f"    {fl:6s} → {tl:6s} : {cnt:5,}  "
                  f"gap: median={np.median(gaps):.2f}s  mean={np.mean(gaps):.2f}s")

    # ── Cut quality ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("CUT-POINT TIER BREAKDOWN")
    print(sep)
    total_cuts = sum(tier_counts.values())
    tier_labels = {
        "long_union_gap":    "1. Long silence",
        "short_union_gap":   "2. Short silence",
        "vad_only_gap":      "3. VAD-only gap",
        "vtc_only_gap":      "4. VTC-only gap",
        "speaker_boundary":  "5. Speaker boundary",
        "hard_cut":          "6. Hard cut",
        "degenerate_window": "7. Degenerate",
    }
    for k, label in tier_labels.items():
        cnt = tier_counts.get(k, 0)
        pct = 100 * cnt / total_cuts if total_cuts > 0 else 0
        print(f"  {label:24s}: {cnt:5,}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':24s}: {total_cuts:5,}")

    # ── Correlation highlights ────────────────────────────────────
    if "correlation" in dfs:
        corr_df = dfs["correlation"]
        metrics = corr_df["metric"].to_list()
        matrix = corr_df.drop("metric").to_numpy()
        print(f"\n{sep}")
        print("TOP CORRELATIONS (|r| > 0.3, off-diagonal)")
        print(sep)
        pairs: list[tuple[float, str, str]] = []
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                r = matrix[i, j]
                if abs(r) > 0.3 and not np.isnan(r):
                    pairs.append((abs(r), metrics[i], metrics[j]))
        pairs.sort(reverse=True)
        for absval, m1, m2 in pairs[:15]:
            r = matrix[metrics.index(m1), metrics.index(m2)]
            print(f"  {m1:25s} ↔ {m2:25s}  r={r:+.3f}")

    print(f"\n{sep}")
    print("END OF DASHBOARD SUMMARY")
    print(sep)
