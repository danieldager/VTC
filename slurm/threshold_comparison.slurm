#!/bin/bash
#SBATCH --job-name=thresh_cmp
#SBATCH --output=logs/compare/thresh_%j.out
#SBATCH --error=logs/compare/thresh_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

DATASET="$1"
shift 1

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch threshold_comparison.slurm <DATASET> [ADDITIONAL_ARGS...]"
    echo "Example: sbatch threshold_comparison.slurm seedlings_10"
    exit 1
fi

echo ""
echo "Threshold Comparison Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export POLARS_SKIP_CPU_CHECK=1

module purge
module load ffmpeg

PYTHONUNBUFFERED=1 \
uv run python -m src.analysis.threshold_comparison "$DATASET" \
    --per_file \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date '+%H:%M:%S')  exit=$EXIT_CODE"
exit $EXIT_CODE
