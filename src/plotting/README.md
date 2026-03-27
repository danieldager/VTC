# Plotting

All dashboard figures are generated from **parquet DataFrames** saved during
the packaging step.  This means **plots can be modified and regenerated
without rerunning the pipeline** — just reload the parquets and call the
dashboard functions.

## Saved data (after a pipeline run)

Everything lives under `output/<dataset>/stats/`:

| File | Granularity | What's inside |
|------|-------------|---------------|
| `clip_stats.parquet` | 1 row / clip | duration, vtc/vad density, IoU, SNR/C50 stats, per-label durations, conversation counts, child fraction, dominant label |
| `segment_stats.parquet` | 1 row / VTC segment | onset, offset, duration, label, SNR/C50 during that segment |
| `turn_stats.parquet` | 1 row / turn | onset, offset, duration, label, segment count, SNR/C50 during the turn |
| `conversation_stats.parquet` | 1 row / conversation | onset, offset, duration, turn count, multi-speaker flag, labels involved, transition count, mean SNR/C50, gap to next conversation |
| `transition_stats.parquet` | 1 row / speaker transition | from→to labels, gap duration, segment durations on each side |
| `file_stats.parquet` | 1 row / source audio file | clip count, total durations (overall + per label), mean SNR/C50, turn/conversation counts |
| `correlation_matrix.parquet` | N × N matrix | Pearson r between all numeric clip-level metrics |
| `tier_counts.json` | 1 dict | Cut-point quality tier counts (for page 4 bar chart) |

Additionally `output/<dataset>/shards/manifest.csv` has one row per clip
with flattened metadata useful for external tools.

## Dashboard pages

All written to `figures/<dataset>/dashboard/`.

### Page 1 — `snr_quality.png` · SNR & Recording Quality (3×3)

1. Per-clip mean SNR histogram
2. SNR vs speech density scatter
3. SNR by dominant label (boxplot)
4. SNR vs VAD–VTC IoU scatter
5. Per-label mean SNR during speech (bar ± std)
6. Intra-clip SNR std histogram
7. Low-SNR fraction by threshold (<0/5/10/15/20 dB)
8. C50 clarity histogram
9. Conversation-level SNR histogram

*Source: `snr_noise.py → save_snr_figures`*

### Page 2 — `conversation_structure.png` · Conversational Structure (3×3)

1. Turn density per clip (turns/min)
2. Speaker transitions (who→who, top 12 pairs)
3. Turns per conversation
4. Response latency by transition type (top 6 pairs)
5. Multi vs single speaker pie chart
6. Conversation duration histogram
7. Turn complexity (single vs multi-segment)
8. Inter-conversation silence histogram
9. SNR vs child fraction scatter

*Source: `speech_turns.py → save_conversation_figures`*

### Page 3 — `turns_conversations.png` · Turns & Conversations (3×3)

1. Turn duration by role — all labels overlaid
2. Turn duration — KCHI only
3. Turn duration — OCH only
4. Turn duration — FEM only
5. Turn duration — MAL only
6. Conversation duration histogram
7. Turns per conversation histogram
8. Inter-conversation duration histogram
9. Conversation SNR & C50 dual histogram

*Source: `speech_turns.py → save_boss_figures`*

### Page 4 — `dataset_overview.png` · Dataset Overview + Cut Quality (3×3)

1. Speech volume by speaker type (hours)
2. Segment duration by label (boxplot)
3. Child speech fraction distribution
4. Cut-point tier breakdown (horizontal bar)
5. Clip duration distribution
6. Clips per source file
7. VAD vs VTC density scatter
8. Speech density per clip
9. Dataset summary text card

*Source: `overview.py → save_overview_figures`*

### Page 5 — `correlation_matrix.png` · Correlation Matrix

Single heatmap of Pearson correlations across all numeric clip metrics.

*Source: `overview.py → save_correlation_figure`*

### Page 6 — `esc_environment.png` · ESC Environment (3×3)

Only rendered when `esc_*` columns are present (i.e. PANNs was run).

1. Dominant ESC type per clip (pie chart)
2. Mean probability per category (bar chart)
3. Probability distribution per category (boxplot)
4. Noise category vs SNR (boxplot)
5. Top 3 ESC types per clip (stacked bar)
6. Noise probability heatmap (top clips)
7. Category probability correlation heatmap
8. Noise-type co-occurrence matrix
9. Per-segment ESC vs duration scatter

*Source: `snr_noise.py → save_noise_figures`*

## Regenerating plots

```python
import polars as pl
from pathlib import Path
from src.plotting.figures import save_all_figures

dataset = "seedlings_10"
stats_dir = Path(f"output/{dataset}/stats")

dfs = {
    name: pl.read_parquet(stats_dir / f"{name}.parquet")
    for name in [
        "clip_stats", "segment_stats", "turn_stats",
        "conversation_stats", "transition_stats",
        "file_stats", "correlation",
    ]
}

import json
tier_counts = json.loads((stats_dir / "tier_counts.json").read_text())

save_all_figures(
    dfs, tier_counts, Path(f"figures/{dataset}/dashboard")
)
```

Run it:

```bash
uv run python -c "
import polars as pl
from pathlib import Path
from src.plotting.figures import save_all_figures

dataset = 'seedlings_10'
stats_dir = Path(f'output/{dataset}/stats')
import json
dfs = {n: pl.read_parquet(stats_dir / f'{n}.parquet')
       for n in ['clip_stats','segment_stats','turn_stats',
                 'conversation_stats','transition_stats',
                 'file_stats','correlation']}
tier_counts = json.loads((stats_dir / 'tier_counts.json').read_text())
save_all_figures(dfs, tier_counts, Path(f'figures/{dataset}/dashboard'))
print('Done — check figures/{dataset}/dashboard/')
"
```

> `tier_counts` is saved as `stats/tier_counts.json` alongside the
> parquets, so all pages regenerate identically.

## Module structure

| Module | Purpose |
|--------|---------|
| `figures.py` | Orchestrator — `save_all_figures` delegates to sub-modules |
| `snr_noise.py` | Pages 1 & 6 — SNR quality + noise environment |
| `speech_turns.py` | Pages 2 & 3 — conversational structure + turns |
| `overview.py` | Pages 4 & 5 — dataset overview + correlation matrix + text summary |
