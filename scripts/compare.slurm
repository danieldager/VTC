#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --output=logs/compare/compare_%j.out
#SBATCH --error=logs/compare/compare_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

DATASET="$1"
shift 1

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch compare.slurm <DATASET> [--manifest PATH] [--path-col COL] [--audio-root DIR]"
    echo "Example: sbatch compare.slurm chunks30"
    exit 1
fi

echo ""
echo "Step 3/3  Compare VAD vs VTC"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export POLARS_SKIP_CPU_CHECK=1

PYTHONUNBUFFERED=1 \
uv run scripts/compare.py "$DATASET" \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date '+%H:%M:%S')  exit=$EXIT_CODE"
exit $EXIT_CODE
