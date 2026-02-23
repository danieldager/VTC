#!/bin/bash
# ==========================================================================
#  Full pipeline: Normalize → Preflight → VAD → VTC → Compare
#
#  All paths are derived from DATASET.  Each step submits a SLURM job with
#  appropriate dependencies so the chain runs end-to-end.
#
#  On the first run with a non-standard manifest (--manifest, --path-col,
#  --audio-root), the pipeline normalizes it into manifests/{DATASET}.csv
#  with absolute paths in a 'path' column.  Subsequent runs need only the
#  dataset name.
#
#  Usage:
#    bash slurm/pipeline.sh [DATASET] [OPTIONS]
#
#  First run (non-standard manifest):
#    bash slurm/pipeline.sh my_data --manifest /data/meta.xlsx \
#        --path-col recording_id --audio-root /store/audio/
#
#  Subsequent runs:
#    bash slurm/pipeline.sh my_data
#
#  Positional:
#    DATASET          Dataset name (default: "chunks30").  Used to derive
#                     output / metadata / figures directories and to locate
#                     the manifest in the manifests/ folder.
#
#  Options:
#    --manifest PATH  Path to a source manifest to normalize.
#    --path-col COL   Column containing audio paths (default: "path").
#    --audio-root DIR Root directory for relative audio paths.
#    --sample N       Process only a random subset of files (for testing).
#                     Pass an integer ≥ 1 for an exact count, or a float
#                     in (0,1) for a fraction.  E.g. --sample 500 or 0.1.
#    --overwrite      Remove all previous outputs for the dataset.
# ==========================================================================

set -euo pipefail

# ---------- Configuration -------------------------------------------------
DATASET="chunks30"
OVERWRITE=false
MANIFEST=""
PATH_COL="path"
AUDIO_ROOT=""
SAMPLE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --path_col)
            PATH_COL="$2"
            shift 2
            ;;
        --audio_root)
            AUDIO_ROOT="$2"
            shift 2
            ;;
        --sample)
            SAMPLE="$2"
            shift 2
            ;;
        --help|-h)
            head -n 38 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage information."
            exit 1
            ;;
        *)
            DATASET="$1"
            shift
            ;;
    esac
done

# VTC array: how many GPU tasks to split inference across
VTC_ARRAY_COUNT=4

# Adaptive thresholding parameters (passed to VTC step)
TARGET_IOU=0.9
THRESHOLD_MIN=0.1
THRESHOLD_STEP=0.1

# Downstream scripts only need --sample (manifest args handled by normalize)
EXTRA_ARGS=""
if [ -n "$SAMPLE" ]; then
    EXTRA_ARGS="--sample $SAMPLE"
fi

# ---------- Overwrite: clear previous run data ----------------------------
if [ "$OVERWRITE" = true ]; then
    echo ""
    echo "Clearing previous data for '$DATASET'"
    rm -rf "output/${DATASET}"
    rm -rf "metadata/${DATASET}"
    rm -rf "figures/${DATASET}"
fi

mkdir -p logs/vad logs/vtc logs/compare

# ==========================================================================
# STEP 0a — Normalize manifest (only when non-standard args are given)
# ==========================================================================
NEEDS_NORMALIZE=false
if [ -n "$MANIFEST" ] || [ "$PATH_COL" != "path" ] || [ -n "$AUDIO_ROOT" ]; then
    NEEDS_NORMALIZE=true
fi

if [ "$NEEDS_NORMALIZE" = true ]; then
    echo ""
    echo "Normalizing manifest..."
    NORM_ARGS="$DATASET"
    [ -n "$MANIFEST" ]       && NORM_ARGS="$NORM_ARGS --manifest $MANIFEST"
    [ "$PATH_COL" != "path" ] && NORM_ARGS="$NORM_ARGS --path-col $PATH_COL"
    [ -n "$AUDIO_ROOT" ]     && NORM_ARGS="$NORM_ARGS --audio-root $AUDIO_ROOT"

    PYTHONPATH="${PYTHONPATH:-}:$(pwd)" uv run python3 src/pipeline/normalize.py $NORM_ARGS
    if [ $? -ne 0 ]; then
        echo "ERROR: Manifest normalization failed." >&2
        exit 1
    fi
else
    # Validate that the manifest exists for this dataset
    PYTHONPATH="${PYTHONPATH:-}:$(pwd)" uv run python3 -c "
from src.utils import resolve_manifest
import sys
try:
    p = resolve_manifest('${DATASET}')
    print(f'  Manifest: {p}')
except (FileNotFoundError, ValueError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

# ==========================================================================
# STEP 0b — Preflight: dataset summary & ETA estimate
# ==========================================================================
echo ""
PYTHONPATH="${PYTHONPATH:-}:$(pwd)" uv run python3 src/pipeline/preflight.py "$DATASET" \
    --vtc-tasks "$VTC_ARRAY_COUNT" \
    --vad-workers 48 \
    $EXTRA_ARGS

echo "Pipeline: $DATASET  sample=${SAMPLE:-all}  GPUs=$VTC_ARRAY_COUNT"
echo ""

# ---------- Step 1: VAD ---------------------------------------------------

VAD_JOB=$(sbatch --parsable \
    slurm/vad.slurm "$DATASET" $EXTRA_ARGS)

echo "  VAD     : $VAD_JOB"

# ---------- Step 2: VTC ---------------------------------------------------

ARRAY_SPEC="0-$((VTC_ARRAY_COUNT - 1))"

VTC_JOB=$(sbatch --parsable \
    --dependency=afterok:${VAD_JOB} \
    --array="${ARRAY_SPEC}" \
    slurm/vtc.slurm "$DATASET" \
        --target_iou "$TARGET_IOU" \
        --threshold_min "$THRESHOLD_MIN" \
        --threshold_step "$THRESHOLD_STEP" \
        $EXTRA_ARGS)

echo "  VTC     : $VTC_JOB  (array ${ARRAY_SPEC})"

# ---------- Step 3: Compare -----------------------------------------------

CMP_JOB=$(sbatch --parsable \
    --dependency=afterok:${VTC_JOB} \
    slurm/compare.slurm "$DATASET")

echo "  Compare : $CMP_JOB"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel $VAD_JOB $VTC_JOB $CMP_JOB"
