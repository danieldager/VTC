#!/bin/bash
#SBATCH --job-name=vad-ten
#SBATCH --output=logs/vad/ten_%j.out
#SBATCH --error=logs/vad/ten_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=cpu,erc-dupoux,gpu-p1,gpu-p2

DATASET="$1"

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch vad.slurm <DATASET>"
    echo "Example: sbatch vad.slurm chunks30"
    exit 1
fi

echo "================================================================"
echo "  STEP 1/3: TenVAD — Voice Activity Detection (CPU)"
echo "================================================================"
echo "  Runs TenVAD on each audio file to detect speech vs silence."
echo "  All paths derived from dataset: $DATASET"
echo "  Outputs:"
echo "    - VAD segments:   output/${DATASET}/vad_raw/ and vad_merged/"
echo "    - Per-file metadata: metadata/${DATASET}/ten/metadata.parquet"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK | Memory: ${SLURM_MEM_PER_NODE}"
echo "Started: $(date)"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/llvm-17.0.6-ibcxgmojxutzsgeoyywyraq5fmvycsfl/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH
export POLARS_SKIP_CPU_CHECK=1

module purge
module load ffmpeg llvm

export PYTHONPATH=$PYTHONPATH:$(pwd)

PYTHONUNBUFFERED=1 \
uv run scripts/vad.py "$DATASET" \
    --workers "$SLURM_CPUS_PER_TASK"

echo ""
echo "VAD completed: $(date)"
