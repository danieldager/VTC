"""Dashboard pages: SNR & Recording Quality, Noise Environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.core import LABEL_COLORS, VTC_LABELS


def _setup():
    """Lazy matplotlib setup, returns plt module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# Colours for the redesigned noise category map.
_NOISE_COLORS: dict[str, str] = {
    "music": "#E24A33",  # red-orange
    "crying": "#348ABD",  # medium blue
    "laughter": "#FBC15E",  # amber yellow
    "singing": "#8EBA42",  # olive green
    "tv_radio": "#988ED5",  # purple
    "vehicle": "#777777",  # gray
    "animal": "#FF8C42",  # orange
    "household": "#C44E52",  # dark red
    "impact": "#8C8C8C",  # medium gray
    "silence": "#64B5CD",  # light blue
    "human_activity": "#FFB347",  # peach
    "nature": "#55A868",  # green
    "machinery": "#A0826D",  # brown
    "alarm_signal": "#CCB974",  # gold
    "environment": "#5B7FA6",  # steel blue
    "other": "#D3D3D3",  # light gray (fallback)
}


def save_snr_figures(
    clip_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    conversation_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """SNR / C50 dashboard — 3×3 panels."""
    plt = _setup()

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("SNR & Recording Quality", fontsize=14, fontweight="bold")

    snr = clip_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
    has_snr = len(snr) > 0

    # 1. Per-clip mean SNR histogram
    ax = axes[0, 0]
    if has_snr:
        ax.hist(snr, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(snr),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(snr):.1f} dB",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Mean SNR per clip (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Per-clip Mean SNR Distribution")

    # 2. SNR vs speech density
    ax = axes[0, 1]
    if has_snr:
        dens = clip_df.filter(pl.col("snr_mean").is_not_null())[
            "vtc_density"
        ].to_numpy()
        snr_f = clip_df.filter(pl.col("snr_mean").is_not_null())["snr_mean"].to_numpy()
        ax.scatter(dens, snr_f, alpha=0.25, s=8, c="#55A868")
    ax.set_xlabel("VTC speech density")
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_title("SNR vs Speech Density")

    # 3. SNR by dominant label (box)
    ax = axes[0, 2]
    if has_snr:
        box_data = []
        labels_used: list[str] = []
        for l in VTC_LABELS:
            vals = clip_df.filter(
                (pl.col("dominant_label") == l) & pl.col("snr_mean").is_not_null()
            )["snr_mean"].to_numpy()
            if len(vals) > 0:
                box_data.append(vals)
                labels_used.append(l)
        if box_data:
            bp = ax.boxplot(box_data, labels=labels_used, patch_artist=True, widths=0.6)
            for patch, l in zip(bp["boxes"], labels_used):
                patch.set_facecolor(LABEL_COLORS.get(l, "#999"))
                patch.set_alpha(0.7)
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_title("SNR by Dominant Label")

    # 4. SNR vs VAD–VTC IoU
    ax = axes[1, 0]
    if has_snr:
        filt = clip_df.filter(pl.col("snr_mean").is_not_null())
        ax.scatter(
            filt["snr_mean"].to_numpy(),
            filt["vad_vtc_iou"].to_numpy(),
            alpha=0.25,
            s=8,
            c="#C44E52",
        )
    ax.set_xlabel("Mean SNR (dB)")
    ax.set_ylabel("VAD–VTC IoU")
    ax.set_title("SNR vs Model Agreement")

    # 5. Per-label mean SNR during speech (bar ± std)
    ax = axes[1, 1]
    if "snr_during" in segment_df.columns:
        seg_snr = segment_df.filter(pl.col("snr_during").is_not_null())
        if len(seg_snr) > 0:
            means, stds, cols = [], [], []
            for l in VTC_LABELS:
                v = seg_snr.filter(pl.col("label") == l)["snr_during"].to_numpy()
                if len(v) > 0:
                    means.append(float(np.mean(v)))
                    stds.append(float(np.std(v)))
                    cols.append(l)
            if cols:
                ax.bar(
                    cols,
                    means,
                    yerr=stds,
                    capsize=4,
                    alpha=0.8,
                    color=[LABEL_COLORS.get(l, "#999") for l in cols],
                    edgecolor="white",
                )
    ax.set_ylabel("Mean SNR during speech (dB)")
    ax.set_title("SNR During Each Speaker Type")

    # 6. Intra-clip SNR std histogram
    ax = axes[1, 2]
    snr_std = clip_df["snr_std"].drop_nulls().drop_nans().to_numpy()
    if len(snr_std) > 0:
        ax.hist(snr_std, bins=50, color="#DD8452", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(snr_std),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(snr_std):.1f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Intra-clip SNR std (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Intra-clip SNR Variability")

    # 7. Low-SNR fraction by threshold
    ax = axes[2, 0]
    if has_snr:
        thresholds = [0, 5, 10, 15, 20]
        fracs = [float((snr < t).mean()) for t in thresholds]
        ax.bar(
            [str(t) for t in thresholds],
            fracs,
            color=["#d62728", "#e74c3c", "#ff7f0e", "#f1c40f", "#2ecc71"],
            edgecolor="white",
            alpha=0.8,
        )
        for i, f in enumerate(fracs):
            ax.text(i, f + 0.01, f"{f:.1%}", ha="center", fontsize=9)
    ax.set_xlabel("SNR threshold (dB)")
    ax.set_ylabel("Fraction of clips below")
    ax.set_title("Low-SNR Clip Fraction")
    ax.set_ylim(0, 1.05)

    # 8. C50 clarity histogram
    ax = axes[2, 1]
    c50 = clip_df["c50_mean"].drop_nulls().drop_nans().to_numpy()
    if len(c50) > 0:
        ax.hist(c50, bins=50, color="#8172B2", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(c50),
            color="red",
            ls="--",
            lw=1,
            label=f"median={np.median(c50):.1f} dB",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Mean C50 per clip (dB)")
    ax.set_ylabel("Count")
    ax.set_title("C50 Clarity Distribution")

    # 9. Conversation-level SNR histogram
    ax = axes[2, 2]
    if "snr_mean" in conversation_df.columns:
        conv_snr = conversation_df["snr_mean"].drop_nulls().drop_nans().to_numpy()
        if len(conv_snr) > 0:
            ax.hist(conv_snr, bins=50, color="#64B5CD", edgecolor="white", alpha=0.8)
            ax.axvline(
                np.median(conv_snr),
                color="red",
                ls="--",
                lw=1,
                label=f"median={np.median(conv_snr):.1f} dB",
            )
            ax.legend(fontsize=8)
    ax.set_xlabel("Mean SNR per conversation (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Conversation-Level SNR")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path}")


# =========================================================================
# Page 2: Conversational Structure (3×3)
# =========================================================================


def _horiz_bars(
    ax,
    labels: list[str],
    values: np.ndarray,
    title: str,
    xlabel: str,
    pct: bool = False,
) -> None:
    """Draw a horizontal bar chart sorted descending, coloured by category."""
    order = np.argsort(values)  # ascending → bottom of chart = largest
    sorted_labels = [labels[i] for i in order]
    sorted_values = values[order]
    colors = [_NOISE_COLORS.get(l, "#D3D3D3") for l in sorted_labels]

    bars = ax.barh(
        sorted_labels, sorted_values, color=colors, edgecolor="white", height=0.7
    )
    for bar, val in zip(bars, sorted_values):
        label_str = (
            f"{val:.1f}%" if pct else f"{val:.3f}" if val < 0.1 else f"{val:.2f}"
        )
        ax.text(
            bar.get_width() + max(sorted_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label_str,
            va="center",
            fontsize=8,
        )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(0, max(sorted_values) * 1.20 if sorted_values.max() > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_noise_figures(
    clip_df: pl.DataFrame,
    segment_df: pl.DataFrame,
    output_path: Path,
    noise_stats_dir: Path | None = None,
) -> None:
    """Noise environment — 3 separate horizontal bar-chart panels.

    If *noise_stats_dir* is provided and contains ``category_stats.parquet``
    (produced by ``src.analysis.noise_stats``), that richer data is used.
    Otherwise falls back to the clip-level ``noise_*`` columns.
    """
    plt = _setup()

    # ── Attempt to load redesigned category stats ─────────────────────────
    cat_stats_path = (
        noise_stats_dir / "category_stats.parquet"
        if noise_stats_dir is not None
        else None
    )
    if cat_stats_path is not None and cat_stats_path.exists():
        cat_df = (
            pl.read_parquet(cat_stats_path)
            .filter(pl.col("map_version") == "new")
            .sort("pw_hours", descending=True)
        )
        categories = cat_df["category"].to_list()
        pw_hours = cat_df["pw_hours"].to_numpy().astype(np.float64)
        mean_prob = cat_df["mean_prob"].to_numpy().astype(np.float64)
        active_rate = cat_df["active_rate"].to_numpy().astype(np.float64)

        fig, axes = plt.subplots(1, 3, figsize=(22, max(6, len(categories) * 0.55 + 2)))
        fig.suptitle(
            "Non-Speech Noise Environment  (PANNs CNN14 — redesigned categories)",
            fontsize=13,
            fontweight="bold",
        )

        _horiz_bars(
            axes[0], categories, pw_hours, "Probability-weighted hours", "Hours"
        )
        _horiz_bars(
            axes[1],
            categories,
            mean_prob,
            "Mean probability\n(across all 1-s windows)",
            "Mean P",
        )
        _horiz_bars(
            axes[2],
            categories,
            active_rate * 100,
            "Active rate\n(fraction of windows with P > 0.05)",
            "% of windows",
            pct=True,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")
        return

    # ── Fallback: clip-level noise_* columns ─────────────────────────────
    noise_cols = [c for c in clip_df.columns if c.startswith("noise_")]
    if not noise_cols:
        return  # no noise data — skip

    cat_names = [c.replace("noise_", "") for c in noise_cols]
    mean_probs = np.array(
        [float(clip_df[c].drop_nulls().mean() or 0) for c in noise_cols]  # type: ignore
    )
    order = np.argsort(mean_probs)[::-1]
    cat_sorted = [cat_names[i] for i in order]
    prob_sorted = mean_probs[order]

    fig, ax = plt.subplots(figsize=(12, max(5, len(cat_names) * 0.55 + 2)))
    fig.suptitle("Noise Environment (PANNs CNN14)", fontsize=13, fontweight="bold")
    _horiz_bars(ax, cat_sorted, prob_sorted, "Mean clip-level probability", "Mean P")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# =========================================================================
# Fine-grained noise pie charts
# =========================================================================


def _category_shades(base_hex: str, n: int) -> list:
    """Return *n* RGBA colours ranging from *base_hex* (darkest) to 50 % lighter."""
    import matplotlib.colors as mcolors

    r0, g0, b0 = mcolors.to_rgb(base_hex)
    if n == 1:
        return [(r0, g0, b0, 1.0)]
    factors = np.linspace(0, 0.55, n)
    return [
        (r0 + (1 - r0) * f, g0 + (1 - g0) * f, b0 + (1 - b0) * f, 1.0) for f in factors
    ]


def save_noise_pie_figures(
    noise_stats_dir: Path,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Big pie charts for fine-grained PANNs noise analysis.

    Produces two pies side-by-side in one figure:
      * Left  — coarse category breakdown (redesigned map)
      * Right — top-*top_n* non-speech AudioSet classes by probability-weighted
                hours, coloured by their parent category; remaining classes
                shown as a single "Everything else" slice.

    Parameters
    ----------
    noise_stats_dir : Path
        Directory containing ``audioset_stats.parquet`` and
        ``category_stats.parquet`` (output of ``src.analysis.noise_stats``).
    output_path : Path
        Destination PNG file.
    top_n : int
        How many AudioSet classes to show individually (default: 20).
    """
    as_path = noise_stats_dir / "audioset_stats.parquet"
    cat_path = noise_stats_dir / "category_stats.parquet"
    if not as_path.exists() or not cat_path.exists():
        return

    plt = _setup()

    as_df = (
        pl.read_parquet(as_path)
        .filter(~pl.col("is_speech"))
        .sort("pw_hours", descending=True)
    )
    cat_df = (
        pl.read_parquet(cat_path)
        .filter(pl.col("map_version") == "new")
        .sort("pw_hours", descending=True)
    )

    # ── Coarse category data ──────────────────────────────────────────────
    coarse_cats = cat_df["category"].to_list()
    coarse_hours = cat_df["pw_hours"].to_numpy().astype(np.float64)
    coarse_colors = [_NOISE_COLORS.get(c, "#D3D3D3") for c in coarse_cats]

    # ── Fine-grained: top-N + "Everything else" ───────────────────────────
    top_df = as_df.head(top_n)
    rest_h = float(as_df.tail(len(as_df) - top_n)["pw_hours"].sum())
    n_rest = len(as_df) - top_n

    fine_names = top_df["class_name"].to_list()
    fine_hours = top_df["pw_hours"].to_numpy().astype(np.float64)
    fine_cats = top_df["new_category"].to_list()

    # Build shaded colours: same-category classes get adjacent shades
    # Sort by (category, pw_hours desc) so same-cat slices sit together.
    group_order: list[tuple[str, str, float]] = sorted(
        zip(fine_cats, fine_names, fine_hours.tolist()),
        key=lambda x: (x[0], -x[2]),
    )
    from collections import Counter

    cat_counts = Counter(c for c, _, _ in group_order)
    cat_shade_pos: dict[str, int] = {}
    fine_colors: list = []
    for cat, name, _ in group_order:
        pos = cat_shade_pos.get(cat, 0)
        shades = _category_shades(_NOISE_COLORS.get(cat, "#D3D3D3"), cat_counts[cat])
        fine_colors.append(shades[pos])
        cat_shade_pos[cat] = pos + 1

    sorted_names = [n for _, n, _ in group_order]
    sorted_hours = np.array([h for _, _, h in group_order])
    sorted_cats_g = [c for c, _, _ in group_order]

    if rest_h > 0:
        sorted_names.append(f"Everything else ({n_rest} classes)")
        sorted_hours = np.append(sorted_hours, rest_h)
        fine_colors.append((0.82, 0.82, 0.82, 1.0))

    total_fine = sorted_hours.sum()
    total_coarse = coarse_hours.sum()

    # ── Layout: left pie (coarse) + right pie (fine) ─────────────────────
    fig = plt.figure(figsize=(26, 13))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.05)
    ax_coarse = fig.add_subplot(gs[0])
    ax_fine = fig.add_subplot(gs[1])

    kw_pie = dict(startangle=90, counterclock=False)

    def _draw_pie(ax, hours, labels, colors, title, total):
        pcts = hours / total * 100
        wedges, _ = ax.pie(
            hours,
            colors=colors,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
            **kw_pie,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

        # Legend: only show slices >= 0.8 %
        legend_entries = [
            (w, f"{lbl}  ({p:.1f}%)")
            for w, lbl, p in zip(wedges, labels, pcts)
            if p >= 0.8
        ]
        if legend_entries:
            leg_wedges, leg_labels = zip(*legend_entries)
            ax.legend(
                leg_wedges,
                leg_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                fontsize=8.5,
                frameon=False,
                handlelength=1.1,
                handleheight=1.1,
            )
        return wedges

    _draw_pie(
        ax_coarse,
        coarse_hours,
        coarse_cats,
        coarse_colors,
        "Non-Speech Noise by Category",
        total_coarse,
    )

    _draw_pie(
        ax_fine,
        sorted_hours,
        sorted_names,
        fine_colors,
        f"Non-Speech Noise by AudioSet Class  (top {top_n})",
        total_fine,
    )

    fig.suptitle(
        "PANNs CNN14 — Probability-weighted hours of non-speech acoustic activity",
        fontsize=12,
        y=0.98,
        color="#444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# =========================================================================
# Convenience: render all pages
# =========================================================================
