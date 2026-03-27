#!/bin/bash
# ==========================================================================
#  Full pipeline: VAD + VTC + SNR + Noise (parallel) → Package
#
#  VAD, VTC, SNR, and Noise are independent and launch simultaneously.
#  Package depends on all four and also runs VAD–VTC comparison internally.
#
#  Preflight auto-detects GPU hardware and dataset stats to set:
#    - VTC batch size (tuned to GPU VRAM)
#    - Array counts for VTC / SNR / Noise (tuned to dataset size + GPU count)
#
#  Both VAD and VTC use a fixed 0.5 sigmoid threshold.
#
#  For threshold-sensitivity analysis, see:
#    bash slurm/threshold_analysis.sh seedlings_1 --sample 0.05
#
#  Usage:
#    bash slurm/pipeline.sh seedlings_1
#    bash slurm/pipeline.sh seedlings_10 --sample 0.5
# ==========================================================================

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

DATASET="${1:?Usage: bash slurm/pipeline.sh DATASET [--sample N]}"
shift

SAMPLE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample) SAMPLE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EXTRA_ARGS=""
[[ -n "$SAMPLE" ]] && EXTRA_ARGS="--sample $SAMPLE"

VTC_THRESHOLD=0.5          # fixed sigmoid threshold — no adaptive sweep

mkdir -p logs/{vad,vtc,snr,esc,package}

# ---------- Preflight: auto-detect resources ------------------------------

echo ""
echo "Running preflight resource detection..."

PREFLIGHT_ENV=$(uv run python -m src.pipeline.preflight "$DATASET" \
    --emit-env $EXTRA_ARGS 2>&1) || {
    echo "WARNING: preflight failed — using conservative defaults"
    PREFLIGHT_ENV=""
}

# Defaults (overridden by preflight if available)
VTC_BATCH_SIZE=128
VTC_ARRAY_COUNT=2
SNR_ARRAY_COUNT=2
ESC_ARRAY_COUNT=2
GPU_NAME="unknown"
GPU_VRAM_GB=0

# Source preflight output
if [[ -n "$PREFLIGHT_ENV" ]]; then
    eval "$PREFLIGHT_ENV"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Full pipeline: $DATASET"
echo "║  sample=${SAMPLE:-all}"
echo "║  GPU: ${GPU_NAME} (${GPU_VRAM_GB} GB VRAM)"
echo "║  VTC: batch=${VTC_BATCH_SIZE}  shards=${VTC_ARRAY_COUNT}"
echo "║  SNR: shards=${SNR_ARRAY_COUNT}  ESC: shards=${ESC_ARRAY_COUNT}"
echo "║  VTC threshold=${VTC_THRESHOLD}  (fixed, no sweep)"
echo "║  VAD + VTC + SNR + ESC run in parallel"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ---------- Steps 1-4: VAD + VTC + SNR + ESC (all parallel) ------------

VAD_JOB=$(sbatch --parsable \
    slurm/vad.slurm "$DATASET" $EXTRA_ARGS)

echo "  1. VAD            : $VAD_JOB"

VTC_ARRAY="0-$((VTC_ARRAY_COUNT - 1))"

VTC_JOB=$(sbatch --parsable \
    --array="${VTC_ARRAY}" \
    slurm/vtc.slurm "$DATASET" \
        --threshold "$VTC_THRESHOLD" \
        --batch_size "$VTC_BATCH_SIZE" \
        $EXTRA_ARGS)

echo "  2. VTC            : $VTC_JOB  (array ${VTC_ARRAY}, batch=${VTC_BATCH_SIZE})"

SNR_ARRAY="0-$((SNR_ARRAY_COUNT - 1))"

SNR_JOB=$(sbatch --parsable \
    --array="${SNR_ARRAY}" \
    slurm/snr.slurm "$DATASET" $EXTRA_ARGS)

echo "  3. SNR (Brouhaha) : $SNR_JOB  (array ${SNR_ARRAY})"

ESC_ARRAY="0-$((ESC_ARRAY_COUNT - 1))"

ESC_JOB=$(sbatch --parsable \
    --array="${ESC_ARRAY}" \
    slurm/esc.slurm "$DATASET" $EXTRA_ARGS)

echo "  4. ESC (PANNs)  : $ESC_JOB  (array ${ESC_ARRAY})"

# ---------- Step 5: Package + Compare + Dashboard (after all) -------------

PKG_JOB=$(sbatch --parsable \
    --dependency=afterok:${VAD_JOB}:${VTC_JOB}:${SNR_JOB}:${ESC_JOB} \
    --job-name=pkg_dash \
    --output=logs/package/pkg_%j.out \
    --error=logs/package/pkg_%j.err \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --partition=erc-dupoux,gpu-p1 \
    --wrap="
        set -euo pipefail
        module purge && module load ffmpeg
        export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:\${LD_LIBRARY_PATH:-}
        export PYTHONPATH=\${PYTHONPATH:-}:\$(pwd)
        export POLARS_SKIP_CPU_CHECK=1

        echo ''
        echo 'Step 5/5  Package + Compare + Dashboard'
        echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        echo \"Job: \$SLURM_JOB_ID  Node: \$(hostname)\"
        echo \"Started: \$(date '+%Y-%m-%d %H:%M:%S')\"
        echo ''

        PYTHONUNBUFFERED=1 \\
        uv run python -m src.pipeline.package $DATASET \\
            $EXTRA_ARGS \\
            --audio_fmt wav \\
            --max_clip 600

        echo ''
        echo \"Completed: \$(date '+%H:%M:%S')\"
    ")

echo "  5. Package+Dash   : $PKG_JOB"

echo ""
echo "Chain: [VAD($VAD_JOB) | VTC($VTC_JOB) | SNR($SNR_JOB) | ESC($ESC_JOB)] → Package($PKG_JOB)"
echo ""
echo "Monitor : squeue -u \$USER"
echo "Cancel  : scancel $VAD_JOB $VTC_JOB $SNR_JOB $ESC_JOB $PKG_JOB"
echo "Pkg log : logs/package/pkg_\${PKG_JOB}.out"
echo ""
