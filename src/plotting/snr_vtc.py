"""Dashboard page: SNR & C50 restricted to VTC speech segments.

Seven panels, 3 top + 4 bottom:

  [0] Per-file mean SNR & C50 histogram
  [1] Per-clip mean SNR & C50 histogram (skips clips with no speech)
  [2] Global statistics info box
  [3] FEM  — per-segment SNR & C50 histogram
  [4] MAL  — per-segment SNR & C50 histogram
  [5] KCHI — per-segment SNR & C50 histogram
  [6] OCH  — per-segment SNR & C50 histogram
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.core import LABEL_COLORS, VTC_LABELS

_ADULT_LABELS = ["FEM", "MAL"]

_SNR_COLOR = "#4C72B0"  # blue
_C50_COLOR = "#DD8452"  # orange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _hist_pair(ax, snr_vals: np.ndarray, c50_vals: np.ndarray) -> None:
    """Overlay SNR and C50 histograms on *ax*, with mean annotations."""
    # Dynamically calculate bins using Sturges' rule based on total sample size
    total_n = len(snr_vals) + len(c50_vals)
    if total_n > 0:
        # Sturges' rule: k = ceil(log2(n) + 1)
        bins = max(8, int(np.ceil(np.log2(total_n) + 1)))
    else:
        bins = 8

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


def _stat_box(ax, rows: list[tuple[bool, str, str]], title: str = "") -> None:
    """Render a structured info card on a hidden axis.

    Each row is ``(is_header, label, value)``.  Headers are rendered in
    grey bold; data rows show label and right-aligned value.
    """
    from matplotlib.patches import FancyBboxPatch

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")

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

    y_top, y_bot = 0.92, 0.08
    step = (y_top - y_bot) / max(len(rows), 1)
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
                0.88,
                y,
                value,
                transform=ax.transAxes,
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
                color="#212529",
            )


def _count_badge(ax, text: str) -> None:
    """Attach a small info badge in the top-right corner of *ax*."""
    ax.text(
        0.97,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#495057",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#f8f9fa",
            edgecolor="#ced4da",
        ),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def save_snr_vtc_figures(
    clip_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    file_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """SNR & C50 during-speech dashboard — 3 top + 4 bottom panels."""
    plt = _setup()
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "SNR & C50 — Brouhaha VAD-Masked (speech frames only)",
        fontsize=14,
        fontweight="bold",
    )

    # 12-column grid: top row = 3 plots × 4 cols; bottom row = 4 plots × 3 cols
    gs = GridSpec(2, 12, figure=fig, hspace=0.45, wspace=1.5)

    ax_file = fig.add_subplot(gs[0, 0:4])
    ax_clip = fig.add_subplot(gs[0, 4:8])
    ax_global = fig.add_subplot(gs[0, 8:12])

    ax_fem = fig.add_subplot(gs[1, 0:3])
    ax_mal = fig.add_subplot(gs[1, 3:6])
    ax_kchi = fig.add_subplot(gs[1, 6:9])
    ax_och = fig.add_subplot(gs[1, 9:12])

    # ------------------------------------------------------------------
    # Aggregate SNR/C50 from segment_df
    # ------------------------------------------------------------------

    # Work only with segments that have at least one measurement
    seg = segment_df.filter(
        pl.col("snr_during").is_not_null() | pl.col("c50_during").is_not_null()
    )

    # Per-clip means across all VTC segments in that clip
    clip_agg = seg.group_by(["uid", "clip_idx"]).agg(
        pl.col("snr_during").mean().alias("snr"),
        pl.col("c50_during").mean().alias("c50"),
    )
    clip_snr = clip_agg["snr"].drop_nulls().to_numpy()
    clip_c50 = clip_agg["c50"].drop_nulls().to_numpy()

    total_clips = len(clip_df)
    included_clips = int(
        clip_agg.filter(
            pl.col("snr").is_not_null() | pl.col("c50").is_not_null()
        ).height
    )
    skipped_clips = total_clips - included_clips

    # Global per-segment values (for overall mean and per-label plots)
    all_snr = seg["snr_during"].drop_nulls().to_numpy()
    all_c50 = seg["c50_during"].drop_nulls().to_numpy()

    # ------------------------------------------------------------------
    # Panel 1 — All segments histogram
    # ------------------------------------------------------------------
    _hist_pair(ax_file, all_snr, all_c50)
    ax_file.set_title(
        "All VTC Segments — SNR & C50 per segment",
        fontsize=10,
    )
    _count_badge(ax_file, f"Segments: {len(all_snr) + len(all_c50):,}")

    # ------------------------------------------------------------------
    # Panel 2 — Per-clip histogram
    # ------------------------------------------------------------------
    _hist_pair(ax_clip, clip_snr, clip_c50)
    ax_clip.set_title(
        "Per-Clip Mean SNR & C50\n(averaged over VTC segments)",
        fontsize=10,
    )
    _count_badge(
        ax_clip,
        f"Total clips:  {total_clips:,}\n"
        f"Included:     {included_clips:,}\n"
        f"Skipped:      {skipped_clips:,}",
    )

    # ------------------------------------------------------------------
    # Panel 3 — Global statistics box
    # ------------------------------------------------------------------
    # Adult speech density per file: (FEM + MAL) / total_file_duration
    adult_densities: np.ndarray = np.array([])
    if len(file_df) > 0 and "total_dur" in file_df.columns:
        fem_col = (
            file_df["total_dur_FEM"]
            if "total_dur_FEM" in file_df.columns
            else pl.Series([0.0] * len(file_df))
        )
        mal_col = (
            file_df["total_dur_MAL"]
            if "total_dur_MAL" in file_df.columns
            else pl.Series([0.0] * len(file_df))
        )
        total_col = file_df["total_dur"]
        mask = total_col > 0
        adult_dur = (fem_col + mal_col).filter(mask)
        total_dur_f = total_col.filter(mask)
        if len(total_dur_f) > 0:
            adult_densities = (adult_dur / total_dur_f).to_numpy()

    # Median duration of adult VTC segments (FEM or MAL)
    adult_segs = segment_df.filter(pl.col("label").is_in(_ADULT_LABELS))
    adult_durs = adult_segs["duration"].to_numpy()

    # Median duration of all VTC segments (all labels)
    all_durs = segment_df["duration"].to_numpy()

    def _fmt(val: float | None, suffix: str) -> str:
        return f"{val:.1f} {suffix}" if val is not None else "—"

    info_rows: list[tuple[bool, str, str]] = [
        (True, "RECORDING QUALITY", ""),
        (
            False,
            "Mean SNR (speech)",
            _fmt(float(np.mean(all_snr)) if len(all_snr) > 0 else None, "dB"),
        ),
        (
            False,
            "Mean C50 (speech)",
            _fmt(float(np.mean(all_c50)) if len(all_c50) > 0 else None, "dB"),
        ),
        (True, "ADULT SPEECH", ""),
        (
            False,
            "Avg adult density",
            (
                f"{float(np.mean(adult_densities)):.1%}"
                if len(adult_densities) > 0
                else "—"
            ),
        ),
        (True, "TURN DURATION", ""),
        (
            False,
            "Median (adult only)",
            f"{float(np.median(adult_durs)):.2f} s" if len(adult_durs) > 0 else "—",
        ),
        (
            False,
            "Median (all labels)",
            f"{float(np.median(all_durs)):.2f} s" if len(all_durs) > 0 else "—",
        ),
    ]
    _stat_box(ax_global, info_rows, title="Global Statistics")

    # ------------------------------------------------------------------
    # Panels 4–7 — Per-label histograms
    # ------------------------------------------------------------------
    label_axes = [ax_fem, ax_mal, ax_kchi, ax_och]
    for ax, lbl in zip(label_axes, VTC_LABELS):
        lseg = segment_df.filter(pl.col("label") == lbl)
        lsnr = lseg["snr_during"].drop_nulls().to_numpy()
        lc50 = lseg["c50_during"].drop_nulls().to_numpy()

        _hist_pair(ax, lsnr, lc50)
        ax.set_title(
            f"{lbl} — SNR & C50 per segment",
            fontsize=10,
            color=LABEL_COLORS.get(lbl, "#333"),
        )
        _count_badge(ax, f"Segments: {len(lseg):,}")

    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")
