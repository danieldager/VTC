#!/bin/bash
# ==========================================================================
#  Comparison pipeline: VAD (with probs) → VTC (full-file, with logits) → Analysis
#
#  Runs on a 10% sample of seedlings to investigate VAD vs VTC disagreement.
#  Both systems save their raw outputs (probs/logits) so we can sweep
#  thresholds offline without re-running inference.
#
#  Usage:
#    bash slurm/comparison.sh                       # defaults: seedlings_10, 10% sample
#    bash slurm/comparison.sh my_dataset 0.05       # custom dataset, 5% sample
#    bash slurm/comparison.sh seedlings_10 0.10 --overwrite
#
#  Positional:
#    DATASET   Dataset name for outputs (default: "seedlings_10")
#    SAMPLE    Fraction of seedlings manifest to use (default: 0.10)
#
#  Options:
#    --overwrite  Clear previous outputs for this dataset before running.
# ==========================================================================

set -euo pipefail

# ---------- Configuration -------------------------------------------------

DATASET="${1:-seedlings_10}"
SAMPLE="${2:-0.10}"
OVERWRITE=false
SOURCE_MANIFEST="seedlings"

# Parse trailing flags
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --help|-h)
            head -n 22 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

# VTC: single GPU (small dataset)
VTC_ARRAY_COUNT=1

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  VAD vs VTC Comparison Pipeline                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Source manifest : ${SOURCE_MANIFEST}"
echo "  Dataset name    : ${DATASET}"
echo "  Sample fraction : ${SAMPLE}"
echo "  VTC GPUs        : ${VTC_ARRAY_COUNT}"
echo "  Overwrite       : ${OVERWRITE}"
echo ""

# ---------- Overwrite: clear previous data --------------------------------
if [ "$OVERWRITE" = true ]; then
    echo "Clearing previous data for '${DATASET}'"
    rm -rf "output/${DATASET}"
    rm -rf "metadata/${DATASET}"
    rm -rf "figures/${DATASET}"
    echo ""
fi

mkdir -p logs/vad logs/vtc logs/compare

# ---------- Create the dataset manifest (symlink to seedlings) ------------
# The pipeline scripts rely on manifests/{DATASET}.csv existing.
# We create a symlink so that --sample draws from the full seedlings manifest.

MANIFEST_SRC="manifests/${SOURCE_MANIFEST}.csv"
MANIFEST_DST="manifests/${DATASET}.csv"

if [ ! -f "$MANIFEST_SRC" ]; then
    echo "ERROR: Source manifest not found: $MANIFEST_SRC"
    exit 1
fi

if [ ! -e "$MANIFEST_DST" ]; then
    ln -s "$(realpath "$MANIFEST_SRC")" "$MANIFEST_DST"
    echo "  Created symlink: ${MANIFEST_DST} → $(realpath "$MANIFEST_SRC")"
fi

# ---------- Preflight check -----------------------------------------------
echo ""
PYTHONPATH="${PYTHONPATH:-}:$(pwd)" uv run python3 src/pipeline/preflight.py "$DATASET" \
    --vtc-tasks "$VTC_ARRAY_COUNT" \
    --vad-workers 48 \
    --sample "$SAMPLE"

echo ""
echo "Pipeline: $DATASET  sample=$SAMPLE  GPUs=$VTC_ARRAY_COUNT"
echo ""

# ---------- Step 1: VAD with --save_logits --------------------------------

VAD_JOB=$(sbatch --parsable \
    slurm/vad.slurm "$DATASET" --save_logits --sample "$SAMPLE")

echo "  1. VAD (save probs) : $VAD_JOB"

# ---------- Step 2: VTC with --save_logits --no_regions -------------------

VTC_JOB=$(sbatch --parsable \
    --dependency=afterok:${VAD_JOB} \
    --array=0 \
    slurm/vtc.slurm "$DATASET" \
        --save_logits \
        --no_regions \
        --sample "$SAMPLE")

echo "  2. VTC (full-file)  : $VTC_JOB"

# ---------- Step 3: Threshold comparison analysis -------------------------

CMP_JOB=$(sbatch --parsable \
    --dependency=afterok:${VTC_JOB} \
    slurm/threshold_comparison.slurm "$DATASET")

echo "  3. Analysis         : $CMP_JOB"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel $VAD_JOB $VTC_JOB $CMP_JOB"
echo ""
echo "Results will be in: figures/${DATASET}/"
echo "  threshold_grid.csv       — aggregate metrics for all combos"
echo "  threshold_heatmap.png    — IoU heatmap"
echo "  volume_sensitivity.png   — speech hours vs threshold"
echo "  per_file_grid.csv        — per-file breakdown"
