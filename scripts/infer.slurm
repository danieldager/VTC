#!/bin/bash
#SBATCH --job-name=vtc_infer
#SBATCH --output=logs/vtc/vtc_infer_%A_%a.out
#SBATCH --error=logs/vtc/vtc_infer_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --array=0

mkdir -p logs/vtc

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch infer.slurm <MANIFEST_FILE> <OUTPUT_DIR> [ADDITIONAL_ARGS...]"
    echo "Example: sbatch infer.slurm manifests/mydataset.parquet output/results --batch_size 256"
    exit 1
fi

MANIFEST="$1"
OUTPUT="$2"
shift 2  # Remove first two arguments, keep the rest

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

echo "VTC Inference Pipeline | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname)"
echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT"
echo "Array Task: ${SLURM_ARRAY_TASK_ID} / $((SLURM_ARRAY_TASK_COUNT-1))"
echo "Started: $(date)"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:$LD_LIBRARY_PATH

module purge
module load ffmpeg
module load git-lfs

echo ""
echo "Running VTC Inference..."
echo ""

PYTHONUNBUFFERED=1 UV_VERBOSE=1 \
uv run scripts/infer.py \
    --manifest "$MANIFEST" \
    --output "$OUTPUT" \
    --array_id "$SLURM_ARRAY_TASK_ID" \
    --array_count "$SLURM_ARRAY_TASK_COUNT" \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"

# # Clean up temporary files on this array task
# if ls /tmp/tmp*.txt 1> /dev/null 2>&1; then
#     rm -f /tmp/tmp*.txt
# fi