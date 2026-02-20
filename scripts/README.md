# Pipeline Reference

End-to-end pipeline for voice activity detection (VAD) and voice type classification (VTC) on long-form audio recordings.

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Data Flow](#3-data-flow)
4. [CLI Reference](#4-cli-reference)
5. [SLURM Reference](#5-slurm-reference)
6. [Core Modules](#6-core-modules)
7. [Utilities](#7-utilities)
8. [Testing](#8-testing)
9. [Failure Analysis (chunks30 Dataset)](#9-failure-analysis-chunks30-dataset)

---

## 1. Quick Start

### One-command pipeline

```bash
# First run — convert a non-standard manifest and run everything:
bash scripts/pipeline.sh my_data \
    --manifest /data/meta.xlsx \
    --path-col recording_id \
    --audio-root /store/audio/

# Subsequent runs — manifest is already normalized:
bash scripts/pipeline.sh my_data
```

This submits three chained SLURM jobs (VAD → VTC → Compare) with automatic dependency handling. Each step waits for the previous one to finish successfully.

### Individual steps

```bash
# VAD only (48-core CPU):
sbatch scripts/vad.slurm my_data

# VTC only (4× GPU array):
sbatch --array=0-3 scripts/vtc.slurm my_data

# Comparison metrics + figures:
sbatch scripts/compare.slurm my_data
```

### Test a small subset first

```bash
bash scripts/pipeline.sh my_data --sample 50
```

---

## 2. Pipeline Overview

The pipeline has three stages, each submitted as a SLURM job:

| Step | Script | Resource | What it does |
|------|--------|----------|--------------|
| **1. VAD** | `vad.py` via `vad.slurm` | 48 CPUs, 128 GB | TenVAD speech detection on every file (multiprocessed) |
| **2. VTC** | `vtc.py` via `vtc.slurm` | 4× GPU (array) | VTC-2.0 inference with adaptive per-file thresholding |
| **3. Compare** | `compare.py` via `compare.slurm` | 8 CPUs, 32 GB | IoU / Precision / Recall metrics, diagnostic figures |

**Key design decisions:**

- **Adaptive thresholding** — VTC reads VAD output to calibrate per-file sigmoid thresholds. Sweeps from 0.5 down to 0.1 and picks the highest threshold where VTC–VAD IoU ≥ 90%. See [Failure Analysis](#9-failure-analysis-chunks30-dataset) for motivation.
- **Activity-region optimization** — For files where VAD detects speech in < 90% of the duration, VTC inference is restricted to speech-active regions only. Cuts GPU time by up to 10× on long recordings with low speech ratios.
- **Checkpoint-based resume** — Both VAD and VTC save periodic checkpoints. If a job is interrupted, re-submitting the same command skips already-completed files.
- **Spawn context for VAD** — Worker processes use `mp.get_context("spawn")` to avoid inheriting torch pages from the parent, keeping per-worker memory at ~31 MB instead of ~500 MB.

---

## 3. Data Flow

All paths are derived from the dataset name. For a dataset called `my_data`:

```
manifests/my_data.csv                  ← normalized manifest (path column)
   │
   ├── STEP 1: vad.py (CPU, 48 workers)
   │   ├── metadata/my_data/metadata.parquet       per-file VAD stats
   │   ├── output/my_data/vad_raw/segments.parquet  raw speech segments
   │   └── output/my_data/vad_merged/segments.parquet  merged segments
   │
   ├── STEP 2: vtc.py (GPU, 4 shards)
   │   ├── output/my_data/vtc_raw/shard_*.parquet   raw VTC segments
   │   ├── output/my_data/vtc_merged/shard_*.parquet merged VTC segments
   │   ├── output/my_data/vtc_meta/shard_*.parquet   per-file threshold/IoU metadata
   │   └── output/my_data/logits/ (optional)         saved raw logits
   │
   └── STEP 3: compare.py (CPU)
       ├── output/my_data/compare_raw.csv           per-file IoU (raw segments)
       ├── output/my_data/compare_merged.csv         per-file IoU (merged segments)
       ├── output/my_data/diagnostics.csv            failure classifications
       ├── figures/my_data/compare_raw.png           6-panel dashboard
       ├── figures/my_data/compare_merged.png        6-panel dashboard
       └── metadata/my_data/metadata.parquet         updated with VTC columns
```

### Manifest format

The pipeline accepts manifests in any of these formats: `.parquet`, `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`, `.jsonl`.

The only requirement is a column containing **absolute paths** to WAV files (16 kHz, mono). All other columns are preserved.

If your manifest uses relative paths or a non-standard column name, use `--manifest`, `--path-col`, and `--audio-root` on first run — the pipeline normalizes it to `manifests/{dataset}.csv` with absolute paths in a `path` column.

---

## 4. CLI Reference

### `pipeline.sh` — Full pipeline orchestrator

```bash
bash scripts/pipeline.sh [DATASET] [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `DATASET` (positional) | `chunks30` | Dataset name — derives all output paths |
| `--manifest PATH` | — | Source manifest to normalize on first run |
| `--path-col COL` | `path` | Column containing audio paths |
| `--audio-root DIR` | — | Root directory for relative paths |
| `--sample N` | — | Random subset (int ≥ 1 = count, float 0–1 = fraction) |
| `--overwrite` | false | Remove all previous outputs before starting |

The script runs normalize and preflight locally, then submits Steps 1–3 as SLURM jobs with `--dependency=afterok` chaining.

---

### `vad.py` — Step 1: Voice Activity Detection

```bash
uv run scripts/vad.py DATASET [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataset` | positional | — | Dataset name |
| `--hop-size` | int | 256 | TenVAD hop size in samples |
| `--threshold` | float | 0.5 | TenVAD speech/silence threshold |
| `-w` / `--workers` | int | all CPUs | Number of parallel workers |
| `--sample` | int/float | — | Process a random subset |

**Outputs:** `metadata/{dataset}/metadata.parquet`, `output/{dataset}/vad_raw/`, `output/{dataset}/vad_merged/`

**Progress logging:** Reports GB-level progress across all in-flight workers every 60 seconds, including partial file progress from each worker via an inter-process queue.

---

### `vtc.py` — Step 2: VTC Inference

```bash
uv run scripts/vtc.py DATASET [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataset` | positional | — | Dataset name |
| `--config` | str | `VTC-2.0/model/config.yml` | Model config path |
| `--checkpoint` | str | `VTC-2.0/model/best.ckpt` | Model checkpoint path |
| `--target_iou` | float | 0.9 | Target VAD–VTC IoU for adaptive thresholding |
| `--threshold_max` | float | 0.5 | Starting (highest) threshold |
| `--threshold_min` | float | 0.1 | Lowest threshold to sweep to |
| `--threshold_step` | float | 0.1 | Step size in sweep |
| `--min_duration_on_s` | float | 0.1 | Remove segments shorter than this |
| `--min_duration_off_s` | float | 0.1 | Fill gaps shorter than this |
| `--batch_size` | int | 128 | GPU batch size |
| `--save_logits` | flag | false | Save raw logits per file |
| `--device` | str | `cuda` | `cuda`, `cpu`, or `mps` |
| `--array_id` | int | — | SLURM array task ID |
| `--array_count` | int | — | Total SLURM array tasks |
| `--sample` | int/float | — | Process a random subset |

**Outputs:** `output/{dataset}/vtc_raw/`, `output/{dataset}/vtc_merged/`, `output/{dataset}/vtc_meta/`

**Progress logging:** Reports files completed and GB-level progress with ETA every 60 seconds or every 5% of files.

---

### `compare.py` — Step 3: VAD vs VTC Comparison

```bash
uv run scripts/compare.py DATASET [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataset` | positional | — | Dataset name |
| `--low-iou-thresh` | float | 0.3 | IoU threshold for failure flagging |

**Outputs:** `output/{dataset}/compare_raw.csv`, `output/{dataset}/compare_merged.csv`, `output/{dataset}/diagnostics.csv`, `figures/{dataset}/compare_*.png`, updated `metadata/{dataset}/metadata.parquet`

---

### `convert.py` — Audio format conversion

```bash
uv run scripts/convert.py --wavs INPUT_DIR --output OUTPUT_DIR [--allow_upsampling]
```

Resamples audio to 16 kHz mono WAV. Use this if your audio is in a different format before running the pipeline.

---

### `normalize.py` — Manifest normalization

```bash
uv run scripts/normalize.py DATASET --manifest PATH [--path-col COL] [--audio-root DIR]
```

Creates a standardized `manifests/{dataset}.csv` with absolute paths. Called automatically by `pipeline.sh` when `--manifest` is provided.

---

### `preflight.py` — Pre-pipeline scan

```bash
uv run scripts/preflight.py DATASET [--vtc-tasks 4] [--vad-workers 48]
```

Reports file count, total size, and estimated wall-clock time. Called automatically by `pipeline.sh`.

---

### `retest.py` — VTC threshold sweep & determinism test

```bash
uv run scripts/retest.py --manifest PATH --output DIR \
    --thresholds 0.1 0.2 0.3 0.4 0.5 --repeats 3
```

Runs VTC on a sample of files at multiple thresholds for debugging. Produces `retest_results.parquet` and `retest_summary.txt`.

---

### `plots.py` — Interactive Plotly dashboard

```bash
uv run scripts/plots.py
```

Generates an interactive HTML dashboard for the chunks5 dataset at `figures/superposition_analysis.html`. Hardcoded paths — primarily for internal analysis.

---

## 5. SLURM Reference

### `vad.slurm` — Step 1 (CPU)

```bash
sbatch scripts/vad.slurm DATASET [EXTRA_ARGS...]
```

| Parameter | Value |
|-----------|-------|
| CPUs | 48 |
| Memory | 128 GB |
| Time limit | 24 h |
| Partition | cpu, erc-dupoux, gpu-p1, gpu-p2 |

Workers set to `$SLURM_CPUS_PER_TASK`. Loads `ffmpeg` and `llvm` modules. Extra arguments forwarded to `vad.py`.

### `vtc.slurm` — Step 2 (GPU array)

```bash
sbatch --array=0-3 scripts/vtc.slurm DATASET [EXTRA_ARGS...]
```

| Parameter | Value |
|-----------|-------|
| GPUs | 1 per task |
| CPUs | 8 per task |
| Memory | 32 GB per task |
| Time limit | 24 h |
| Array | 0–3 (4 shards) |
| Partition | erc-dupoux, gpu-p1, gpu-p2 |

Each array task processes ~25% of the manifest. Loads `ffmpeg` and `git-lfs` modules.

### `compare.slurm` — Step 3 (CPU)

```bash
sbatch scripts/compare.slurm DATASET [EXTRA_ARGS...]
```

| Parameter | Value |
|-----------|-------|
| CPUs | 8 |
| Memory | 32 GB |
| Time limit | 3 h |
| Partition | erc-dupoux, gpu-p1, gpu-p2 |

### `test.slurm` — Test suite runner

```bash
sbatch scripts/test.slurm [PYTEST_ARGS...]
```

Runs the full 82-test pytest suite on a compute node (required for TenVAD tests that need `libc++.so.1`). Extra arguments forwarded to pytest.

---

## 6. Core Modules

The `scripts/core/` package contains reusable, testable components shared across CLI scripts.

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `intervals.py` | Pure interval arithmetic | `merge_pairs`, `total_duration`, `compute_iou`, `intervals_to_segments` |
| `regions.py` | Activity-region VTC optimization | `merge_into_activity_regions`, `activity_region_coverage`, `forward_pass_regions`, `forward_pass_full_file` |
| `thresholds.py` | Adaptive & default threshold sweeping | `find_best_threshold_regions`, `apply_default_threshold_regions` |
| `vad_processing.py` | Per-file VAD (runs in workers) | `process_file`, `get_runs`, `runs_to_segments`, `resample_block` |
| `parallel.py` | Parallel VAD driver | `run_vad_parallel` (spawn context, GB-level progress queue) |
| `checkpoint.py` | Resumable job checkpoints | `save_checkpoint`, `clear_checkpoint` |
| `metadata.py` | VTC metadata constructors | `vtc_meta_row`, `vtc_error_row`, `load_vad_reference` |

---

## 7. Utilities

`scripts/utils.py` provides shared helpers used across all CLI scripts:

| Function | Description |
|----------|-------------|
| `get_dataset_paths(name)` | Returns a `DatasetPaths` dataclass with `.manifest`, `.output`, `.metadata`, `.figures` |
| `load_manifest(path)` | Reads any supported manifest format into a Polars DataFrame |
| `sample_manifest(df, n)` | Random sub-sample (int = count, float = fraction) |
| `shard_list(items, id, count)` | Split items into contiguous SLURM array shards |
| `merge_segments_df(df, off, on)` | Collar-based gap filling + minimum duration filter |
| `atomic_write_parquet(df, path)` | Write via temp-file + atomic rename |
| `log_benchmark(step, ...)` | Append timing record to `logs/benchmarks.jsonl` |
| `resolve_manifest(arg)` | Auto-detect manifest from bare name or explicit path |

---

## 8. Testing

The test suite covers all core modules with 82 tests across 7 files.

### Running tests

```bash
# Login node (71/82 pass — some TenVAD tests skipped):
uv run python3 -m pytest tests/

# Compute node via SLURM (all 82 pass):
sbatch scripts/test.slurm

# Stop on first failure:
sbatch scripts/test.slurm -x
```

### Test files

| File | Tests | Coverage |
|------|------:|----------|
| `test_intervals.py` | ~20 | `merge_pairs`, `total_duration`, `compute_iou`, `intervals_to_pairs` |
| `test_regions.py` | ~12 | `merge_into_activity_regions`, `activity_region_coverage` |
| `test_checkpoint.py` | ~5 | `save_checkpoint`, `clear_checkpoint` |
| `test_metadata.py` | ~7 | `vtc_error_row`, `vtc_meta_row`, `load_vad_reference` |
| `test_parallel.py` | ~3 | `run_vad_parallel` with spawn context (requires TenVAD) |
| `test_vad_processing.py` | ~15 | `get_runs`, `runs_to_segments`, `process_file` (requires TenVAD) |
| `test_stitched_audio.py` | ~10 | End-to-end VAD on stitched long-form audio fixtures |

### Notes

- Tests requiring TenVAD auto-skip on login nodes where `libc++.so.1` is unavailable.
- Tests requiring `torchcodec` auto-skip when the package is not installed.
- The `test.slurm` script loads `ffmpeg` and `llvm` modules and adds the required `LD_LIBRARY_PATH` entries.

---

## 9. Failure Analysis (chunks30 Dataset)

> Date: 2026-02-15 · Dataset: `manifests/chunks30.parquet` (128,913 files, 308 audiobooks)

### Summary

The VTC model produces **empty or near-empty detections** for a significant minority of audio files. Comparison with TenVAD (which detects speech reliably across the same files) reveals two distinct failure modes.

### Overall numbers

| Metric | Count | % of total |
|--------|------:|----------:|
| Total files compared | 127,971 | 100% |
| IoU = 0 (no overlap) | 8,980 | 7.0% |
| IoU < 0.3 (poor) | 28,789 | 22.5% |
| IoU ≥ 0.8 (good) | 56,992 | 44.5% |
| Empty RTTMs (0 bytes) | 8,856 | 6.9% |
| vtc_silent (VTC found zero speech) | 8,633 | — |
| vtc_under_detect (VTC found <30% of VAD) | 19,856 | — |

Mean IoU across all files: **0.612** (std 0.323).

### Failure Mode 1: Book-level model blindness

The failures **cluster strongly by audiobook** (i.e. by narrator). Of the 204 books with >100 chunks:

- **46 books** have mean IoU < 0.3
- **81 books** have mean IoU < 0.5
- **61 books** have mean IoU > 0.8

Within the worst-affected books, the pattern is nearly all-or-nothing. For example, `8698_LibriVox_en` (499 chunks) has 392 completely silent chunks and a mean IoU of 0.017.

#### Top 10 worst-performing books

| Book ID | Mean IoU | Chunks |
|---------|----------|--------|
| 8051_LibriVox_en | 0.002 | 42 |
| 8043_LibriVox_en | 0.004 | 59 |
| 5339_LibriVox_en | 0.008 | 41 |
| 8698_LibriVox_en | 0.017 | 499 |
| 4386_LibriVox_en | 0.020 | 84 |
| 4290_LibriVox_en | 0.023 | 170 |
| 4868_LibriVox_en | 0.029 | 192 |
| 861_LibriVox_en  | 0.038 | 246 |
| 7532_LibriVox_en | 0.043 | 1035 |
| 4818_LibriVox_en | 0.058 | 937 |

#### Likely cause

The VTC model (VTC-2.0, `surgical_hubert_hydra`) was trained on child-directed speech data with labels `KCHI`, `OCH`, `MAL`, `FEM`. Audiobook narrators whose voice characteristics don't match the training distribution fail to activate any speaker class above the default 0.5 sigmoid threshold.

#### Evidence ruling out bugs

- All 128,913 files have RTTM files on disk (no missing files)
- Audio format is identical across all files: `pcm_s16le`, 16 kHz, mono
- All 4 SLURM shards completed successfully with exit code 0 and zero errors
- No errors or warnings in any VTC log file
- The same books that fail here are the same books with low IoU — consistent, not random

### Failure Mode 2: Short-file edge case

Even in well-performing books (mean IoU > 0.5), files shorter than the model's **4-second chunk duration** have elevated failure rates:

| Duration bracket | Files | vtc_dur=0 | Empty % | Mean IoU |
|-----------------|------:|----------:|--------:|---------:|
| < 4s | 11,493 | 1,068 | 9.3% | 0.721 |
| 4–10s | 20,933 | 444 | 2.1% | 0.757 |
| 10–30s | 31,885 | 98 | 0.3% | 0.763 |
| 30–60s | 12,939 | 3 | 0.0% | 0.755 |
| > 60s | 11,538 | 0 | 0.0% | 0.773 |

Files under 4s fall through to a tail-frame fallback path in segma's `apply_model_on_audio()` which yields lower quality results.

### Non-determinism investigation → RESOLVED

A retest (`scripts/retest.py`) ran 96 sample files 3× each and compared against the original pipeline:

- **Within a single session**: perfectly deterministic — 0/96 files showed any variation across 3 repeats.
- **Across sessions**: small boundary effects — 15/79 matched files differed by >0.1s from the original pipeline (max diff = 0.59s, mean = 0.054s). Consistent with CUDA autotuning / floating-point order-of-operations differences across GPU types.

**Conclusion**: Deterministic within a run. Cross-session floating-point variation is NOT the cause of the large-scale failures.

### Threshold sweep results

Lowering the sigmoid threshold from the default 0.5 dramatically recovers "silent" files:

| Threshold | Files detected | Mean speech (s) |
|-----------|:--------------:|:---------------:|
| 0.10      | 95/96          | 23.5            |
| 0.20      | 93/96          | 17.1            |
| 0.30      | 88/96          | 13.6            |
| 0.40      | 77/96          | 11.2            |
| 0.50      | 67/96          | 9.3             |

#### By category

| Category | At thresh=0.5 | At thresh=0.1 | Mean dur change |
|----------|:------------:|:------------:|:---------------:|
| vtc_silent (30 files) | 1/30 | 29/30 | 0.0s → 4.2s |
| vtc_under_detect (47 files) | 47/47 (4.6s avg) | 47/47 (29.9s avg) | +550% |
| good_baseline (18 files) | 18/18 (37.7s avg) | 18/18 (40.0s avg) | +6% (stable) |

#### Logit analysis

Raw logit inspection confirms the model IS seeing speech patterns — just below the 0.5 threshold:

- Many "silent" files have **MAL** activations at 0.30–0.49 across the majority of frames
- Some have **FEM** activations near threshold
- Good baseline files show near-maximum activations (0.95+), so threshold changes barely affect them (+6%)

The model assigns partial confidence to out-of-distribution speech — enough to detect at a lower threshold, but not enough to breach 0.5 for any of its four classes.

### Solution: Adaptive per-file thresholding

Since TenVAD reliably detects speech across all files, we use it as ground truth to calibrate VTC thresholds per file:

1. VTC runs a single GPU forward pass producing raw logits
2. For each file, it starts at threshold 0.5 and steps down until VTC–VAD IoU ≥ target (default 90%)
3. The highest threshold meeting the target is selected (preserves precision)
4. Per-file threshold and IoU are recorded in `output/{dataset}/vtc_meta/`

This is built into `vtc.py` and runs automatically when VAD output is available. 