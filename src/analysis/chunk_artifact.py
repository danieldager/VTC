"""
Chunk Boundary Artifact Analysis
=================================

VTC (Segma) processes audio in 4-second chunks of exactly 64,000 samples at
16 kHz.  Each chunk produces 199 output frames spaced 320 samples (20 ms)
apart.  The *chunk step* — the offset between the starts of consecutive chunks
— is 199 × 320 = 63,680 samples = 3.98 s.  A segment that spans all 199 frames
of one chunk lasts exactly:

    (199 - 1) × 320  +  (rf_size - 1)  +  1
  = 198 × 320 + 399 + 1
  = 63,760 samples  =  3.985 s

This script investigates whether those processing boundaries create an
over-representation of VTC segment durations at multiples of 3.985 s.

Figures produced
----------------
fig1  Turn duration histogram (raw VTC) with chunk-multiple markers
fig2  Onset-phase density: onset_s % chunk_step_s             (cyclic plot)
fig3  Duration spectrum: fine-grained histogram 0–20 s showing discrete spikes
fig4  Raw vs merged comparison: do the spikes survive the 0.3 s collar merge?

Usage
-----
    python -m src.analysis.chunk_artifact [dataset]   # default: seedlings_10
"""

from __future__ import annotations

import argparse
from math import prod, floor
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Model constants (HuBERT surgical_hydra)
# ---------------------------------------------------------------------------

_KERNELS = (10, 3, 3, 3, 3, 2, 2)
_STRIDES = (5, 2, 2, 2, 2, 2, 2)
_PADDINGS = (0, 0, 0, 0, 0, 0, 0)
_SR = 16_000
_CHUNK_S = 4.0
_CHUNK_F = int(_CHUNK_S * _SR)  # 64,000 samples

# Derived
_RF_STEP: int = prod(_STRIDES)  # 320 samples = 20 ms
_P0: int = sum(
    _PADDINGS[i] * prod(_STRIDES[:i]) for i in range(len(_PADDINGS))
)  # 0 for this model (no padding)
_RT: int = sum(
    (1 + _PADDINGS[i] - _KERNELS[i]) * prod(_STRIDES[:i]) for i in range(len(_KERNELS))
)  # −399 for this model

_HAS_EVEN = any(k % 2 == 0 for k in _KERNELS)
_N_WINDOWS: int = (
    floor(
        (
            _CHUNK_F
            - (sum((k - 1) * prod(_STRIDES[:i]) for i, k in enumerate(_KERNELS)) + 1)
        )
        / (_RF_STEP + (1 if _HAS_EVEN else 0))
    )
    + 1
)  # = 199

# Chunk step: how far apart consecutive chunk starts are (in samples / seconds)
CHUNK_STEP_F: int = _N_WINDOWS * _RF_STEP  # 63,680 samples
CHUNK_STEP_S: float = CHUNK_STEP_F / _SR  # 3.98 s

# Duration of a full-chunk segment (all N_WINDOWS frames active)
FULL_CHUNK_DUR_F: int = (_N_WINDOWS - 1) * _RF_STEP - _RT + 1  # 63,760 samples
FULL_CHUNK_DUR_S: float = FULL_CHUNK_DUR_F / _SR  # 3.985 s


def peak_seconds(n: int) -> float:
    """Predicted segment duration for a run of n consecutive full chunks."""
    total_frames = n * _N_WINDOWS
    return ((total_frames - 1) * _RF_STEP - _RT + 1) / _SR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_dir(path: Path) -> pl.DataFrame:
    parquets = sorted(path.glob("shard_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet shards in {path}")
    return pl.concat([pl.read_parquet(p) for p in parquets])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _lazy_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig1_duration_histogram(
    raw_df: pl.DataFrame, out: Path, merged_df: pl.DataFrame | None = None
) -> None:
    """Duration histogram 0–20 s with n-chunk markers; raw and merged side-by-side.

    Most raw VTC segments are very short (< 100 ms) because the model emits a
    positive detection on any individual 20 ms output frame that exceeds the
    threshold — no minimum duration is enforced before the merge step.  The log
    y-axis makes both the short-segment bulk and the chunk-multiple spikes visible
    at the same time.
    """
    plt = _lazy_plt()

    dfs = [raw_df] + ([merged_df] if merged_df is not None else [])
    panel_labels = [f"Raw VTC  ({len(raw_df):,} segments)"] + (
        [f"Merged VTC  ({len(merged_df):,} segments)"] if merged_df is not None else []
    )

    fig, axes = plt.subplots(1, len(dfs), figsize=(10 * len(dfs), 5), sharey=False)
    if len(dfs) == 1:
        axes = [axes]

    bins = np.arange(0, 20.05, 0.05).tolist()  # 50 ms bins
    peak_colors = ["#d62728", "#e377c2", "#bcbd22", "#17becf", "#8c564b"]

    for ax, df, panel_label in zip(axes, dfs, panel_labels):
        d = df["duration"].to_numpy()
        d = d[d <= 20.0]
        frac_short = (d < 0.1).mean()

        ax.hist(d, bins=bins, color="#4C72B0", alpha=0.85, linewidth=0)
        ax.set_yscale("log")

        for n in range(1, 6):
            p = peak_seconds(n)
            if p > 20:
                break
            ax.axvline(
                p, color=peak_colors[n - 1], lw=1.5, ls="--", label=f"n={n}: {p:.3f} s"
            )

        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(
            f"{panel_label}\n{100 * frac_short:.0f}% of segments shorter than 100 ms"
        )
        ax.legend(fontsize=9, ncol=5)

    fig.suptitle(
        "VTC segment durations — 50 ms bins, log y-axis\n"
        f"Dashed lines: predicted n-chunk peaks  "
        f"(chunk step = {CHUNK_STEP_S:.3f} s,  full-chunk dur = {FULL_CHUNK_DUR_S:.3f} s)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "fig1_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  fig1 → {out / 'fig1_histogram.png'}")


def fig2_onset_phase(raw_df: pl.DataFrame, out: Path) -> None:
    """Cyclic onset-phase density: onset % chunk_step_s.

    For each segment we compute where in the 3.98 s chunk cycle its onset falls
    (onset modulo chunk_step).  If the model had no boundary artifact, onsets
    would be uniformly distributed across the cycle — the histogram would be flat.
    A spike near 0 means that many segments start right at a chunk boundary.

    The right panel filters to segments lasting exactly one processing window
    (~3.985 s).  These are segments where the model classified ALL 199 output
    frames of a single chunk as positive.  Their onset spike at 0 shows that
    these specifically tend to start at chunk boundaries — i.e. the model misses
    the true speech onset in the previous chunk and only picks it up from frame 0
    of the next one.
    """
    plt = _lazy_plt()

    onsets = raw_df["onset"].to_numpy() % CHUNK_STEP_S

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Onset phase relative to chunk cycle  (chunk_step = {CHUNK_STEP_S:.3f} s)\n"
        "Flat = no artifact.  Spike near 0 = onsets cluster at chunk boundaries.",
        fontsize=11,
    )

    bins = np.linspace(0, CHUNK_STEP_S, 80)

    # Left: all segments
    ax = axes[0]
    counts, edges = np.histogram(onsets, bins=bins)
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        color="#4C72B0",
        alpha=0.8,
        label="observed",
    )
    ax.axhline(
        counts.mean(),
        color="red",
        ls="--",
        lw=1.5,
        label=f"uniform mean ({int(counts.mean()):,})",
    )
    ax.axvline(0, color="#d62728", lw=2, alpha=0.7, label="chunk boundary (t=0)")
    ax.set_yscale("log")
    ax.set_xlabel("onset % chunk_step (s)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("All raw segments")
    ax.legend(fontsize=9)

    # Right: only segments lasting exactly one processing window (dur ≈ 3.985 s)
    # "One processing window" = all 199 output frames of a single chunk are active.
    ax = axes[1]
    mask = np.abs(raw_df["duration"].to_numpy() - FULL_CHUNK_DUR_S) <= 0.02
    onsets1 = raw_df["onset"].to_numpy()[mask] % CHUNK_STEP_S
    counts1, _ = np.histogram(onsets1, bins=bins)
    ax.bar(
        edges[:-1],
        counts1,
        width=np.diff(edges),
        color="#e377c2",
        alpha=0.85,
        label="observed",
    )
    ax.axhline(
        counts1.mean(),
        color="red",
        ls="--",
        lw=1.5,
        label=f"uniform mean ({int(counts1.mean()):,})",
    )
    ax.axvline(0, color="#d62728", lw=2, alpha=0.7, label="chunk boundary")
    ax.set_yscale("log")
    ax.set_xlabel("onset % chunk_step (s)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(
        f"Segments lasting exactly one processing window (~{FULL_CHUNK_DUR_S:.3f} s)\n"
        f"{mask.sum():,} of {len(raw_df):,} total raw segments"
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "fig2_onset_phase.png", dpi=150)
    plt.close(fig)
    print(f"  fig2 → {out / 'fig2_onset_phase.png'}")


def fig3_duration_spectrum(raw_df: pl.DataFrame, out: Path) -> None:
    """Duration spectrum 0–25 s — 50 ms bins, log y-axis, discrete spikes at chunk multiples.

    Why multiples?  When a speaker talks continuously through n consecutive chunk
    boundaries without a gap in detection, all n×199 output frames are active and
    the segment duration is exactly n × 3.985 s.  Each higher multiple is rarer
    because it requires the model to sustain full-chunk detection across more
    consecutive windows.
    """
    plt = _lazy_plt()

    durations = raw_df["duration"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "VTC segment duration spectrum — predicted chunk-multiple peaks marked\n"
        "Spikes at n × 3.985 s arise when n consecutive processing windows are fully active",
        fontsize=12,
        fontweight="bold",
    )

    for ax_i, (ax, xlim) in enumerate(zip(axes, [(0, 12), (0, 25)])):
        mask = durations <= xlim[1]
        bins = np.arange(0, xlim[1] + 0.05, 0.05)
        counts, edges = np.histogram(durations[mask], bins=bins)

        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            alpha=0.7,
            color="#4C72B0",
            linewidth=0,
            label="observed",
        )
        ax.set_yscale("log")

        # Predicted peaks
        n = 1
        while True:
            p = peak_seconds(n)
            if p > xlim[1]:
                break
            ax.axvline(
                p,
                color="#d62728",
                lw=1.3,
                ls="--",
                alpha=0.8,
                label=f"n={n} ({p:.3f} s)" if ax_i == 0 else None,
            )
            n += 1

        # Annotate enrichment ratio at each peak (on log scale: position text by multiplying)
        if ax_i == 0:
            baseline = np.median(counts[counts > 0])
            n = 1
            while True:
                p = peak_seconds(n)
                if p > xlim[1]:
                    break
                idx = int(p / 0.05)
                if idx < len(counts) and counts[idx] > 0:
                    ratio = counts[idx] / max(baseline, 1)
                    ax.text(
                        p + 0.1,
                        counts[idx] * 2.0,
                        f"{ratio:.0f}×",
                        color="#d62728",
                        fontsize=7,
                        va="bottom",
                    )
                n += 1

        ax.set_xlim(*xlim)
        ax.set_xlabel("Segment duration (s)")
        ax.set_ylabel("Count (log scale)")
        if ax_i == 0:
            ax.legend(fontsize=8, ncol=6)

    fig.tight_layout()
    fig.savefig(out / "fig3_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"  fig3 → {out / 'fig3_spectrum.png'}")


def fig4_raw_vs_merged(
    raw_df: pl.DataFrame, merged_df: pl.DataFrame, out: Path
) -> None:
    """Compare duration distributions before and after 0.3 s collar merge."""
    plt = _lazy_plt()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    fig.suptitle(
        "Chunk artifact: raw VTC vs merged (0.3 s collar)\n"
        "Dashed red: predicted n-chunk peak positions",
        fontsize=12,
        fontweight="bold",
    )

    for ax, df, title in zip(
        axes,
        [raw_df, merged_df],
        [
            f"Raw VTC  ({len(raw_df):,} segments)",
            f"Merged VTC  ({len(merged_df):,} segments)",
        ],
    ):
        d = df["duration"].to_numpy()
        d = d[d <= 20.0]
        bins = np.arange(0, 20.05, 0.05).tolist()  # 50 ms bins for readability
        ax.hist(d, bins=bins, color="#4C72B0", alpha=0.85, linewidth=0)
        ax.set_yscale("log")

        for n in range(1, 6):
            p = peak_seconds(n)
            if p > 20:
                break
            ax.axvline(
                p,
                color="#d62728",
                lw=1.5,
                ls="--",
                label=f"n={n}: {p:.3f} s" if n <= 3 else None,
            )

        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "fig4_raw_vs_merged.png", dpi=150)
    plt.close(fig)
    print(f"  fig4 → {out / 'fig4_raw_vs_merged.png'}")


def fig5_per_label(raw_df: pl.DataFrame, out: Path) -> None:
    """Per-label duration histogram to check if all labels show the artifact."""
    plt = _lazy_plt()

    from src.core import LABEL_COLORS, VTC_LABELS

    labels = [l for l in VTC_LABELS if l in raw_df["label"].unique().to_list()]
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), sharey=False)
    if len(labels) == 1:
        axes = [axes]
    fig.suptitle(
        "Chunk artifact by speaker label (raw VTC, 20 ms bins)",
        fontsize=12,
        fontweight="bold",
    )

    bins = np.arange(0, 12.01, 0.02).tolist()

    for ax, lbl in zip(axes, labels):
        d = raw_df.filter(pl.col("label") == lbl)["duration"].to_numpy()
        d = d[d <= 12.0]
        color = LABEL_COLORS.get(lbl, "#4C72B0")
        ax.hist(d, bins=bins, color=color, alpha=0.85, linewidth=0)
        ax.set_yscale("log")

        for n in range(1, 4):
            p = peak_seconds(n)
            if p > 12:
                break
            ax.axvline(p, color="#d62728", lw=1.3, ls="--", label=f"n={n}: {p:.2f} s")

        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(f"{lbl}  (n={len(d):,})")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig5_per_label.png", dpi=150)
    plt.close(fig)
    print(f"  fig5 → {out / 'fig5_per_label.png'}")


def print_summary(raw_df: pl.DataFrame) -> None:
    """Print key statistics to the console."""
    total = len(raw_df)
    print(f"\n{'='*60}")
    print("CHUNK BOUNDARY ARTIFACT SUMMARY")
    print(f"{'='*60}")
    print(f"Chunk step        : {CHUNK_STEP_F} samples = {CHUNK_STEP_S:.4f} s")
    print(f"Full-chunk dur    : {FULL_CHUNK_DUR_F} samples = {FULL_CHUNK_DUR_S:.4f} s")
    print(f"Total segments    : {total:,}")
    print()
    print(
        f"{'n':>3}  {'predicted (s)':>14}  {'count ±10ms':>11}  {'% of total':>10}  {'enrichment':>10}"
    )
    print("-" * 56)

    # Enrichment baseline: expected count per 20 ms bin if uniform
    # Over 20s range with 1000 bins of 20ms, expected per bin ≈ total * 0.02 / median_dur_range
    max_dur: float = raw_df["duration"].cast(pl.Float64).max() or 1.0  # type: ignore[assignment]
    baseline_per_20ms = total * 0.02 / max_dur

    for n in range(1, 11):
        p = peak_seconds(n)
        if p > max_dur:
            break
        c = raw_df.filter(
            (pl.col("duration") > p - 0.01) & (pl.col("duration") < p + 0.01)
        ).height
        pct = 100 * c / total
        enrichment = c / max(baseline_per_20ms, 1)
        print(f"{n:>3}  {p:>14.5f}  {c:>11,}  {pct:>9.2f}%  {enrichment:>9.0f}×")

    print()
    # Onset phase enrichment
    chunk_step = CHUNK_STEP_S
    onsets_mod = raw_df["onset"].to_numpy() % chunk_step
    window = 0.040  # ±20 ms window
    near_start = (onsets_mod < window / 2).sum()
    near_end = (onsets_mod > chunk_step - window / 2).sum()
    expected = total * (window / 2) / chunk_step
    print(
        f"Onset-phase enrichment at chunk START: "
        f"{near_start:,}  (×{near_start/expected:.1f} expected)"
    )
    print(
        f"Onset-phase enrichment at chunk END  : "
        f"{near_end:,}  (×{near_end/expected:.1f} expected)"
    )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk boundary artifact analysis")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="seedlings_10",
        help="Dataset name (default: seedlings_10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save figures (default: figures/{dataset}/chunk_artifact/)",
    )
    args = parser.parse_args()

    base = Path("output") / args.dataset
    raw_path = base / "vtc_raw"
    merged_path = base / "vtc_merged"

    out = args.output_dir or Path("figures") / args.dataset / "chunk_artifact"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw VTC segments from {raw_path} …")
    raw_df = _load_dir(raw_path)
    print(f"  {len(raw_df):,} segments, {raw_df['uid'].n_unique()} files")

    merged_df: pl.DataFrame | None = None
    if merged_path.exists():
        print(f"Loading merged VTC segments from {merged_path} …")
        merged_df = _load_dir(merged_path)
        print(f"  {len(merged_df):,} segments")

    print_summary(raw_df)

    print("Generating figures …")
    fig1_duration_histogram(raw_df, out, merged_df)
    fig2_onset_phase(raw_df, out)
    fig3_duration_spectrum(raw_df, out)
    if merged_df is not None:
        fig4_raw_vs_merged(raw_df, merged_df, out)
    fig5_per_label(raw_df, out)

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
