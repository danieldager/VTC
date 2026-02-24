#!/bin/bash
set -euo pipefail

#
# End-to-end packaging test on 1% of seedlings.
#
# Runs: VAD (t=0.50) + VTC (t=0.50, no_regions) → package → listener
#
# Usage:
#   bash slurm/package_test.sh
#   bash slurm/package_test.sh --force   # wipe outputs and re-run from scratch
#

DATASET="seedlings_1"
SAMPLE="0.01"
THRESHOLD="0.50"

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
    shift
fi

cd "$(dirname "$0")/.." || exit 1

if $FORCE; then
    echo ""
    echo "Deleting output/${DATASET} and metadata/${DATASET} ..."
    rm -rf "output/${DATASET}" "metadata/${DATASET}"
fi
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Packaging test — 1% seedlings @ t=$THRESHOLD  ║"
echo "╚══════════════════════════════════════════╝"
echo "  dataset  : $DATASET"
echo "  sample   : $SAMPLE"
echo "  threshold: $THRESHOLD"
echo ""

# ---------- Manifest symlink ---------------------------------------------
MANIFEST_SRC="manifests/seedlings.csv"
MANIFEST_DST="manifests/${DATASET}.csv"

if [ ! -e "$MANIFEST_SRC" ]; then
    echo "ERROR: $MANIFEST_SRC not found" >&2
    exit 1
fi

if [ ! -e "$MANIFEST_DST" ]; then
    ln -s "$(realpath "$MANIFEST_SRC")" "$MANIFEST_DST"
    echo "  Created symlink: ${MANIFEST_DST} → $(realpath "$MANIFEST_SRC")"
fi

mkdir -p logs/package

# ---------- Step 1a: VAD (CPU) --------------------------------------------

VAD_JOB=$(sbatch --parsable \
    slurm/vad.slurm "$DATASET" \
        --threshold "$THRESHOLD" \
        --sample "$SAMPLE")

echo "  1a. VAD (t=$THRESHOLD)    : $VAD_JOB"

# ---------- Step 1b: VTC (GPU, single task) --------------------------------

VTC_JOB=$(sbatch --parsable \
    --array=0 \
    slurm/vtc.slurm "$DATASET" \
        --no_regions \
        --threshold "$THRESHOLD" \
        --sample "$SAMPLE")

echo "  1b. VTC (no_regions)  : $VTC_JOB"

# ---------- Step 2: Package -----------------------------------------------

PKG_JOB=$(sbatch --parsable \
    --dependency=afterok:${VAD_JOB}:${VTC_JOB} \
    --job-name=package \
    --output=logs/package/pkg_%j.out \
    --error=logs/package/pkg_%j.err \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --partition=cpu,erc-dupoux,gpu-p1,gpu-p2 \
    --wrap="
        set -euo pipefail
        module purge && module load ffmpeg
        export PYTHONPATH=\${PYTHONPATH:-}:\$(pwd)
        export POLARS_SKIP_CPU_CHECK=1

        echo 'Package pipeline'
        echo '━━━━━━━━━━━━━━━━━━'
        echo \"Job: \$SLURM_JOB_ID  Node: \$(hostname)\"
        echo \"Started: \$(date '+%Y-%m-%d %H:%M:%S')\"

        PYTHONUNBUFFERED=1 \\
        uv run python -m src.pipeline.package $DATASET \\
            --sample $SAMPLE \\
            --audio_fmt flac \\
            --buffer 5 \\
            --max_gap 10 \\
            --min_seg 0.5 \\
            --max_clip 600

        echo ''
        echo 'Extracting sample clips for validation...'
        PYTHONUNBUFFERED=1 \\
        uv run python -m src.packaging.listener \\
            output/$DATASET/shards \\
            -n 50 --seed 42 --wav

        echo ''
        echo \"Completed: \$(date '+%H:%M:%S')\"
    ")

echo "  2.  Package + Listen  : $PKG_JOB"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel $VAD_JOB $VTC_JOB $PKG_JOB"
echo ""
echo "Results:"
echo "  Shards  : output/$DATASET/shards/"
echo "  Samples : output/$DATASET/shards/samples/"
echo "  Logs    : logs/package/pkg_\${PKG_JOB}.out"
echo ""
