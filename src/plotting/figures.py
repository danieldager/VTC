"""Figure orchestrator — renders all plot pages from cached DataFrames.

Sub-modules
-----------
- ``snr_noise``     — SNR & Recording Quality, Noise Environment
- ``speech_turns``  — Conversational Structure, Turns & Conversations
- ``overview``      — Dataset Overview, Correlation, Text Summary
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.plotting.overview import (
    print_dataset_summary,
    save_correlation_figure,
    save_overview_figures,
)
from src.plotting.snr_noise import (
    save_noise_figures,
    save_noise_pie_figures,
    save_snr_figures,
)
from src.plotting.snr_vtc import save_snr_vtc_figures
from src.plotting.speech_turns import save_boss_figures, save_conversation_figures


def save_all_figures(
    dfs: dict[str, pl.DataFrame],
    tier_counts: dict[str, int],
    fig_dir: Path,
    noise_stats_dir: Path | None = None,
) -> None:
    """Render every figure page from cached DataFrames.

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
    noise_stats_dir : Path | None
        Optional path to ``noise_stats/`` directory produced by
        ``src.analysis.noise_stats``.  When provided, enables richer
        noise figures.
    """
    # Sub-directories by information type
    snr_dir = fig_dir / "snr"
    speech_dir = fig_dir / "speech"
    overview_dir = fig_dir / "overview"
    noise_dir = fig_dir / "noise"
    for d in (snr_dir, speech_dir, overview_dir):
        d.mkdir(parents=True, exist_ok=True)

    clip_df = dfs["clip_stats"]
    segment_df = dfs["segment_stats"]
    turn_df = dfs["turn_stats"]
    conv_df = dfs["conversation_stats"]
    trans_df = dfs["transition_stats"]
    file_df = dfs["file_stats"]

    # snr/quality.png — SNR & Recording Quality
    save_snr_figures(clip_df, segment_df, conv_df, snr_dir / "quality.png")

    # snr/vtc_segments.png — SNR & C50 during VTC speech segments only
    save_snr_vtc_figures(clip_df, segment_df, file_df, snr_dir / "vtc_segments.png")

    # speech/conversation_structure.png — Conversational Structure
    save_conversation_figures(
        clip_df,
        turn_df,
        conv_df,
        trans_df,
        segment_df,
        speech_dir / "conversation_structure.png",
    )

    # speech/turns.png — Turns & Conversations
    save_boss_figures(turn_df, conv_df, speech_dir / "turns.png")

    # overview/dataset.png — Dataset Overview + Cut Quality (combined 3×3)
    save_overview_figures(
        clip_df, file_df, segment_df, tier_counts, overview_dir / "dataset.png"
    )

    # overview/correlation.png — Correlation Matrix
    if "correlation" in dfs:
        save_correlation_figure(dfs["correlation"], overview_dir / "correlation.png")

    # noise/ — Noise Environment (PANNs)
    noise_cols = [c for c in clip_df.columns if c.startswith("noise_")]
    n_pages = 5 + (1 if "correlation" in dfs else 0)
    has_noise_stats = (
        noise_stats_dir is not None
        and (noise_stats_dir / "category_stats.parquet").exists()
    )
    if noise_cols or has_noise_stats:
        noise_dir.mkdir(parents=True, exist_ok=True)
        save_noise_figures(
            clip_df,
            segment_df,
            noise_dir / "environment.png",
            noise_stats_dir=noise_stats_dir,
        )
        n_pages += 1
    if has_noise_stats:
        save_noise_pie_figures(
            noise_stats_dir,  # type: ignore
            noise_dir / "categories.png",
        )
        n_pages += 1

    print(f"\n  Figures: {fig_dir}/ ({n_pages} pages)")

    # Print text summary for log parsing
    print_dataset_summary(dfs, tier_counts)
