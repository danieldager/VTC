#!/bin/bash
#SBATCH --job-name=vtc_infer
#SBATCH --output=logs/vtc/infer_%A_%a.out
#SBATCH --error=logs/vtc/infer_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

DATASET="$1"
shift 1  # remaining args passed through to vtc.py

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch vtc.slurm <DATASET> [ADDITIONAL_ARGS...]"
    echo "Example: sbatch vtc.slurm chunks30 --target_iou 0.85"
    echo "Example: sbatch vtc.slurm my_data --manifest /data/meta.xlsx --path-col recording_id"
    exit 1
fi

echo ""
echo "Step 2/3  VTC Inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: $(hostname)"
echo "Shard: ${SLURM_ARRAY_TASK_ID}/$((SLURM_ARRAY_TASK_COUNT-1))"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
export POLARS_SKIP_CPU_CHECK=1

module purge
module load ffmpeg
module load git-lfs

echo ""

PYTHONUNBUFFERED=1 \
uv run src/pipeline/vtc.py "$DATASET" \
    --array_id "$SLURM_ARRAY_TASK_ID" \
    --array_count "$SLURM_ARRAY_TASK_COUNT" \
    "$@"

EXIT_CODE=$?

echo ""
echo "Completed: $(date '+%H:%M:%S')  exit=$EXIT_CODE"
exit $EXIT_CODE
