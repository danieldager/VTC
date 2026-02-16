#!/bin/bash
# ==========================================================================
#  Full pipeline: VAD → VTC (with adaptive thresholding) → Compare
#
#  All paths are derived from DATASET.  Each step submits a SLURM job with
#  appropriate dependencies so the chain runs end-to-end.
#
#  Usage:  bash scripts/pipeline.sh [DATASET] [--overwrite]
#          Default DATASET is "chunks30".
#          --overwrite removes all previous outputs for the dataset.
# ==========================================================================

set -euo pipefail

# ---------- Configuration -------------------------------------------------
DATASET="chunks30"
OVERWRITE=false

for arg in "$@"; do
    case "$arg" in
        --overwrite) OVERWRITE=true ;;
        *)           DATASET="$arg" ;;
    esac
done

# VTC array: how many GPU tasks to split inference across
VTC_ARRAY_COUNT=3

# Adaptive thresholding parameters (passed to VTC step)
TARGET_IOU=0.9
THRESHOLD_MIN=0.1
THRESHOLD_STEP=0.1

# Derived paths (must match scripts/utils.py get_dataset_paths)
MANIFEST="manifests/${DATASET}.parquet"

# ---------- Sanity checks -------------------------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

# ---------- Overwrite: clear previous run data ----------------------------
if [ "$OVERWRITE" = true ]; then
    echo ""
    echo "  --overwrite: removing previous data for '$DATASET'"
    rm -rf "output/${DATASET}"
    rm -rf "metadata/${DATASET}"
    rm -rf "figures/${DATASET}"
    echo "  Cleared: output/ metadata/ figures/"
fi

mkdir -p logs/vad logs/vtc logs/compare

echo "============================================================"
echo "  Pipeline: $DATASET"
echo "  Manifest: $MANIFEST"
echo "  VTC tasks: $VTC_ARRAY_COUNT"
echo "  Target IoU: $TARGET_IOU"
echo "============================================================"

# ==========================================================================
# STEP 1 — TenVAD  (CPU)
# ==========================================================================
echo ""
echo "[Step 1/3] Submitting TenVAD job..."

VAD_JOB=$(sbatch --parsable \
    scripts/vad.slurm "$DATASET")

echo "  VAD job: $VAD_JOB"

# ==========================================================================
# STEP 2 — VTC Inference + adaptive thresholding (GPU array, depends on VAD)
# ==========================================================================
echo ""
echo "[Step 2/3] Submitting VTC inference array (${VTC_ARRAY_COUNT} tasks)..."

ARRAY_SPEC="0-$((VTC_ARRAY_COUNT - 1))"

VTC_JOB=$(sbatch --parsable \
    --dependency=afterok:${VAD_JOB} \
    --array="${ARRAY_SPEC}" \
    scripts/infer.slurm "$DATASET" \
        --target_iou "$TARGET_IOU" \
        --threshold_min "$THRESHOLD_MIN" \
        --threshold_step "$THRESHOLD_STEP")

echo "  VTC job: $VTC_JOB  (array: ${ARRAY_SPEC})"

# ==========================================================================
# STEP 3 — Compare VAD vs VTC + merge metadata (CPU, depends on VTC)
# ==========================================================================
echo ""
echo "[Step 3/3] Submitting compare job..."

CMP_JOB=$(sbatch --parsable \
    --dependency=afterok:${VTC_JOB} \
    scripts/compare.slurm "$DATASET")

echo "  Compare job: $CMP_JOB"

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "============================================================"
echo "  All jobs submitted.  Dependency chain:"
echo "    VAD     : $VAD_JOB"
echo "    VTC     : $VTC_JOB  (array ${ARRAY_SPEC})"
echo "    Compare : $CMP_JOB"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Cancel:  scancel $VAD_JOB $VTC_JOB $CMP_JOB"
echo "============================================================"
