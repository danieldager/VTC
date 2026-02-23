#!/bin/bash
#SBATCH --job-name=vtc_tests
#SBATCH --output=logs/tests/pytest_%j.out
#SBATCH --error=logs/tests/pytest_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

export PYTHONPATH=$PYTHONPATH:$(pwd)
export POLARS_SKIP_CPU_CHECK=1

module purge
module load ffmpeg llvm

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/llvm-17.0.6-ibcxgmojxutzsgeoyywyraq5fmvycsfl/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH

echo ""
PYTHONUNBUFFERED=1 uv run python3 -m pytest tests/ -v -rA --tb=short --durations=10 "$@"

EXIT_CODE=$?
echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
