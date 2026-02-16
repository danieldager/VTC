#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --output=logs/compare/compare_%j.out
#SBATCH --error=logs/compare/compare_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu,erc-dupoux,gpu-p1,gpu-p2

DATASET="$1"
shift 1

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch compare.slurm <DATASET>"
    echo "Example: sbatch compare.slurm chunks30"
    exit 1
fi

echo "================================================================"
echo "  STEP 3/3: Compare — VAD vs VTC Evaluation"
echo "================================================================"
echo "  Computes per-file IoU / Precision / Recall between VAD and"
echo "  VTC outputs.  Also merges VTC per-file metadata into the"
echo "  VAD metadata file."
echo "  All paths derived from dataset: $DATASET"
echo "  Outputs:"
echo "    - Metrics:  output/${DATASET}/compare_raw.csv, compare_merged.csv"
echo "    - Diagnostics: output/${DATASET}/diagnostics.csv"
echo "    - Figures:  figures/${DATASET}/compare_*.png"
echo "    - Updated:  metadata/${DATASET}/ten/metadata.parquet"
echo "================================================================"
echo "Job: $SLURM_JOB_ID | Node: $(hostname)"
echo "Started: $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)

PYTHONUNBUFFERED=1 \
uv run scripts/compare.py "$DATASET" \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
