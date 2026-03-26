"""
Plotting for VAD coverage analysis results.

Creates visualizations showing missed speech if only using VAD instead of full VTC.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

logger = logging.getLogger(__name__)

# Speaker type color scheme
LABEL_COLORS = {
    "KCHI": "#1f77b4",  # blue
    "OCH": "#ff7f0e",  # orange
    "MAL": "#2ca02c",  # green
    "FEM": "#d62728",  # red
}


def plot_vad_coverage(csv_path: Path, output_dir: Path | None = None):
    """
    Create plots from VAD coverage analysis CSV.

    Args:
        csv_path: Path to vad_coverage_analysis.csv
        output_dir: Output directory for plots. Defaults to same dir as CSV.
    """
    logger.info(f"Loading results from {csv_path}")
    df = pl.read_csv(csv_path)

    if output_dir is None:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse rows into a keyed dict
    results = {}
    for row in df.iter_rows(named=True):
        results[row["metric"]] = row

    labels = ["KCHI", "OCH", "MAL", "FEM"]
    colors = [LABEL_COLORS[label] for label in labels]

    def get(metric: str, label: str, cast=float) -> float:
        return cast(results[metric][label])

    total_segments = {l: get("Total VTC segments", l, int) for l in labels}
    cut_segments = {l: get("VTC segments with >=1 cut", l, int) for l in labels}
    missed_duration = {
        l: get("Total missed speech (seconds)", l, float) for l in labels
    }
    missed_pct = {
        l: float(results["% VTC speech missed"][l].rstrip("%")) for l in labels
    }

    # Create 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "VAD Coverage Analysis: Missed Speech if Using VAD-Only Pipeline",
        fontsize=14,
        fontweight="bold",
    )

    def bar_with_labels(ax, values, fmt="{:,}", y_label="", title="", ylim=None):
        ax.bar(
            labels, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
        )
        ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if ylim:
            ax.set_ylim(ylim)
        for i, v in enumerate(values):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontweight="bold")

    # Plot 1: VTC segments with >=1 cut
    bar_with_labels(
        axes[0, 0],
        [cut_segments[l] for l in labels],
        y_label="Number of Segments",
        title="VTC Segments with ≥1 Cut",
    )

    # Plot 2: % of VTC segments affected
    pcts = [
        (cut_segments[l] / total_segments[l] * 100) if total_segments[l] > 0 else 0
        for l in labels
    ]
    bar_with_labels(
        axes[0, 1],
        pcts,
        fmt="{:.1f}%",
        y_label="Percentage (%)",
        title="% of VTC Segments with ≥1 Cut",
        ylim=[0, 105],
    )

    # Plot 3: % of total VTC speech duration missed
    bar_with_labels(
        axes[1, 0],
        [missed_pct[l] for l in labels],
        fmt="{:.1f}%",
        y_label="% of Total VTC Duration",
        title="% of VTC Speech Duration Missed by VAD",
        ylim=[0, max(missed_pct.values()) * 1.15],
    )

    # Plot 4: Missed speech duration in hours
    hours = [missed_duration[l] / 3600 for l in labels]
    bar_with_labels(
        axes[1, 1],
        hours,
        fmt="{:.1f}h",
        y_label="Duration (Hours)",
        title="Speech Duration Missed by VAD (exact)",
    )

    plt.tight_layout()

    output_png = output_dir / "coverage.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_png}")

    total_missed_hours = sum(missed_duration.values()) / 3600
    logger.info(
        f"Total speech missed: {total_missed_hours:.1f} hours | "
        f"Affected segments: {sum(cut_segments.values()):,}"
    )

    return fig


def main(dataset: str = "seedlings_10", output_dir: Path | None = None):
    """
    Main entry point for plotting VAD coverage results.

    Args:
        dataset: Dataset name (used to locate results).
        output_dir: Optional output directory. Defaults to output/{dataset}.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Paths
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "output" / dataset / "vad_coverage_analysis.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Analysis CSV not found: {csv_path}. Run vad_coverage_analysis.py first."
        )

    if output_dir is None:
        output_dir = project_root / "figures" / dataset / "vad"

    plot_vad_coverage(csv_path, output_dir)


if __name__ == "__main__":
    main()
