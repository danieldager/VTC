"""Figure orchestrator — renders master dashboards from cached DataFrames.

Delegates to ``master.py`` which produces two 3×3 summary PNGs:
  1. Dataset & Content Overview
  2. Recording & Conversation Quality
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.plotting.master import (
    print_dataset_summary,
    save_master_overview,
    save_master_quality,
)


def save_all_figures(
    dfs: dict[str, pl.DataFrame],
    tier_counts: dict[str, int],
    fig_dir: Path,
    esc_stats_dir: Path | None = None,
) -> None:
    """Render master dashboard figures from cached DataFrames.

    Parameters
    ----------
    dfs : dict[str, pl.DataFrame]
        Output of ``save_all_stats``.  Expected keys:
        clip_stats, segment_stats, turn_stats, conversation_stats,
        transition_stats, file_stats, correlation.
    tier_counts : dict[str, int]
        Cut-tier breakdown from ``build_clips``.
    fig_dir : Path
        Root figure directory.  Pages are saved as PNG files inside it.
    esc_stats_dir : Path | None
        Optional path to ``esc_stats/`` directory produced by
        ``src.analysis.esc_stats``.  When provided, enables richer
        ESC figures.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    clip_df = dfs["clip_stats"]
    segment_df = dfs["segment_stats"]
    turn_df = dfs["turn_stats"]
    conv_df = dfs["conversation_stats"]
    trans_df = dfs["transition_stats"]
    file_df = dfs["file_stats"]

    # Master 1 — Dataset & Content Overview
    save_master_overview(
        clip_df, segment_df, file_df, tier_counts,
        fig_dir / "master_overview.png",
    )

    # Master 2 — Recording & Conversation Quality
    save_master_quality(
        clip_df, segment_df, turn_df, conv_df, trans_df,
        fig_dir / "master_quality.png",
        esc_stats_dir=esc_stats_dir,
    )

    print(f"\n  Figures: {fig_dir}/ (2 master pages)")

    # Print text summary for log parsing
    print_dataset_summary(dfs, tier_counts)
