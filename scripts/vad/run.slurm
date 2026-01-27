#!/bin/bash
#SBATCH --job-name=vad-pyannote
#SBATCH --output=logs/vad/pyannote_%A_%a.out
#SBATCH --error=logs/vad/pyannote_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --array=0

if [ -z "$1" ]; then
    echo "Usage: sbatch vad.slurm <MANIFEST_FILE> "
    echo "Example: sbatch vad.slurm manifests/mydataset.txt"
    exit 1
fi

MANIFEST="$1"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:$LD_LIBRARY_PATH

module purge
module load ffmpeg

# Use .venv directly to skip uv checks
# PYTHON=".venv/bin/python"
# if [ ! -x "$PYTHON" ]; then
#     echo "Error: Virtual environment python not found at $PYTHON"
#     exit 1
# fi

echo "VAD Pipeline | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Manifest: $MANIFEST"
echo "Array Task: ${SLURM_ARRAY_TASK_ID} / $((SLURM_ARRAY_TASK_COUNT - 1))"
echo "Started: $(date)"

# $PYTHON scripts/pyannote.py "$MANIFEST" \
uv run --frozen scripts/vad/pyannote.py "$MANIFEST" \
    --array-id "$SLURM_ARRAY_TASK_ID" \
    --array-count "$SLURM_ARRAY_TASK_COUNT"

EXIT_CODE=$?

echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
