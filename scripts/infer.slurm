#!/bin/bash
#SBATCH --job-name=vtc_infer
#SBATCH --output=logs/vtc/infer_%A_%a.out
#SBATCH --error=logs/vtc/infer_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

DATASET="$1"
shift 1  # remaining args passed through to infer.py

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch infer.slurm <DATASET> [ADDITIONAL_ARGS...]"
    echo "Example: sbatch infer.slurm chunks30 --target_iou 0.85"
    exit 1
fi

echo "================================================================"
echo "  STEP 2/3: VTC — Voice Type Classification (GPU)"
echo "================================================================"
echo "  Runs segma model on each audio file, classifying speech as"
echo "  KCHI/OCH/MAL/FEM.  Adaptive thresholding uses VAD reference"
echo "  to find the optimal per-file sigmoid threshold."
echo "  All paths derived from dataset: $DATASET"
echo "  Outputs:"
echo "    - VTC segments: output/${DATASET}/vtc_raw/ and vtc_merged/"
echo "    - Per-file meta: output/${DATASET}/vtc_meta/"
echo "================================================================"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Node: $(hostname)"
echo "Array Task: ${SLURM_ARRAY_TASK_ID} / $((SLURM_ARRAY_TASK_COUNT-1))"
echo "Started: $(date)"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

module purge
module load ffmpeg
module load git-lfs

echo ""

PYTHONUNBUFFERED=1 \
uv run scripts/infer.py "$DATASET" \
    --array_id "$SLURM_ARRAY_TASK_ID" \
    --array_count "$SLURM_ARRAY_TASK_COUNT" \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
