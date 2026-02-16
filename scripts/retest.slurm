#!/bin/bash
#SBATCH --job-name=vtc_retest
#SBATCH --output=logs/vtc/retest_%j.out
#SBATCH --error=logs/vtc/retest_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2
#SBATCH --gres=gpu:1

# Usage: sbatch scripts/retest.slurm [ADDITIONAL_ARGS...]
# Example:
#   sbatch scripts/retest.slurm
#   sbatch scripts/retest.slurm --thresholds 0.05 0.1 0.2 0.3 0.4 0.5 --repeats 5

echo "VTC Retest | Job: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "Started: $(date)"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

module purge
module load ffmpeg
module load git-lfs

MANIFEST="manifests/retest_sample.parquet"
OUTPUT="output/retest"

echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT"
echo ""

PYTHONUNBUFFERED=1 \
uv run scripts/retest.py \
    --manifest "$MANIFEST" \
    --output "$OUTPUT" \
    --thresholds 0.1 0.2 0.3 0.4 0.5 \
    --repeats 3 \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
