#!/bin/bash
# ==========================================================================
#  Re-package + VTC-on-clips + alignment analysis
#
#  Assumes VAD, VTC, SNR, and Noise outputs already exist.
#  Only re-runs packaging (with grid-snapped clip boundaries),
#  then VTC on the new clips, then the alignment comparison.
#
#  Usage:
#    bash slurm/repackage_test.sh seedlings_10
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

DATASET="${1:?Usage: bash slurm/repackage_test.sh DATASET}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Re-package + VTC-on-clips + Alignment: $DATASET"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ---------- Step 1: Re-package (CPU only) ---------------------------------
# Remove old shards and stats so package.py regenerates them
rm -rf "output/${DATASET}/shards" "output/${DATASET}/stats" 2>/dev/null || true
mkdir -p "output/${DATASET}"

PKG_JOB=$(sbatch --parsable \
    --job-name=repack \
    --output="logs/package/repack_%j.out" \
    --error="logs/package/repack_%j.err" \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --partition=erc-dupoux,gpu-p1 \
    --wrap="
        set -euo pipefail
        cd $(pwd)
        module purge && module load ffmpeg
        export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:\${LD_LIBRARY_PATH:-}
        export PYTHONPATH=\${PYTHONPATH:-}:\$(pwd)
        export POLARS_SKIP_CPU_CHECK=1

        echo 'Step 1: Re-packaging with grid-snapped boundaries'
        echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        echo \"Job: \$SLURM_JOB_ID  Node: \$(hostname)\"
        echo \"Started: \$(date '+%Y-%m-%d %H:%M:%S')\"

        PYTHONUNBUFFERED=1 \\
        uv run python -m src.pipeline.package $DATASET \\
            --audio_fmt wav \\
            --max_clip 600

        echo \"Completed: \$(date '+%H:%M:%S')\"
    ")
echo "  1. Package        : $PKG_JOB"

# ---------- Step 2: VTC on clips (GPU, after packaging) -------------------
# Clear old VTC clip results
rm -rf "output/${DATASET}/vtc_clips"

VTC_CLIPS_JOB=$(sbatch --parsable \
    --dependency=afterok:${PKG_JOB} \
    --array=0-5 \
    slurm/vtc_clips.slurm "$DATASET")
echo "  2. VTC on clips   : $VTC_CLIPS_JOB (array 0-5)"

# ---------- Step 3: Alignment analysis (CPU, after VTC clips) -------------
ALIGN_JOB=$(sbatch --parsable \
    --dependency=afterok:${VTC_CLIPS_JOB} \
    --job-name=align \
    --output="logs/tests/align_%j.out" \
    --error="logs/tests/align_%j.err" \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --partition=erc-dupoux,gpu-p1 \
    --wrap="
        set -euo pipefail
        cd $(pwd)
        module purge
        export PYTHONPATH=\${PYTHONPATH:-}:\$(pwd)
        export POLARS_SKIP_CPU_CHECK=1

        echo 'Step 3: Alignment analysis'
        echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        echo \"Job: \$SLURM_JOB_ID  Node: \$(hostname)\"
        echo \"Started: \$(date '+%Y-%m-%d %H:%M:%S')\"

        PYTHONUNBUFFERED=1 \\
        uv run python -m src.analysis.vtc_clip_alignment $DATASET

        echo \"Completed: \$(date '+%H:%M:%S')\"
    ")
echo "  3. Alignment      : $ALIGN_JOB"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Package log:   logs/package/repack_${PKG_JOB}.out"
echo "VTC clips log: logs/vtc/vtc_clips_${VTC_CLIPS_JOB}_*.out"
echo "Alignment log: logs/tests/align_${ALIGN_JOB}.out"
echo ""
echo "After all jobs finish, check results:"
echo "  cat logs/tests/align_${ALIGN_JOB}.out"
echo "  ls figures/${DATASET}/vtc_clip_alignment_*.png"
