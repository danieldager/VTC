# VTC Detection Failure Analysis

Date: 2026-02-15
Dataset: `manifests/chunks30.parquet` (128,913 files, 308 audiobooks)

## Summary

The VTC (Voice Type Classification) model produces **empty or near-empty detections** for a significant minority of audio files. Comparison with TenVAD (which detects speech reliably across the same files) reveals two distinct failure modes.

## Overall numbers

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

## Failure Mode 1: Book-level model blindness

The failures **cluster strongly by audiobook** (i.e. by narrator). Of the 204 books with >100 chunks:

- **46 books** have mean IoU < 0.3
- **81 books** have mean IoU < 0.5
- **61 books** have mean IoU > 0.8

Within the worst-affected books, the pattern is nearly all-or-nothing. For example, `8698_LibriVox_en` (499 chunks) has 392 completely silent chunks and a mean IoU of 0.017.

### Top 10 worst-performing books

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

### Likely cause

The VTC model (VTC-2.0, `surgical_hubert_hydra`) was trained on child-directed speech data with labels `KCHI`, `OCH`, `MAL`, `FEM`. Audiobook narrators whose voice characteristics don't match the training distribution fail to activate any speaker class above the default 0.5 sigmoid threshold. The rare detections in failing books are tiny spurious activations (e.g., 0.06s classified as KCHI in a book with only adult male narration).

### Evidence ruling out bugs

- All 128,913 files have RTTM files on disk (no missing files)
- Audio format is identical across all files: `pcm_s16le`, 16 kHz, mono
- All 4 SLURM shards completed successfully with exit code 0 and zero errors
- No errors or warnings in any VTC log file
- The same books that fail here are the same books with low IoU — consistent, not random

## Failure Mode 2: Short-file edge case

Even in well-performing books (mean IoU > 0.5), files shorter than the model's **4-second chunk duration** have elevated failure rates:

| Duration bracket | Files | vtc_dur=0 | Empty % | Mean IoU |
|-----------------|------:|----------:|--------:|---------:|
| < 4s | 11,493 | 1,068 | 9.3% | 0.721 |
| 4–10s | 20,933 | 444 | 2.1% | 0.757 |
| 10–30s | 31,885 | 98 | 0.3% | 0.763 |
| 30–60s | 12,939 | 3 | 0.0% | 0.755 |
| > 60s | 11,538 | 0 | 0.0% | 0.773 |

Files under 4s fall through to a tail-frame fallback path in segma's `apply_model_on_audio()` which yields lower quality results.

## Open question: Non-determinism → RESOLVED

Previous runs on the same data reportedly had a smaller proportion of low-IoU files. A retest (`scripts/retest_thresholds.py`) ran 96 sample files 3× each and compared against the original pipeline:

- **Within a single session**: perfectly deterministic — 0/96 files showed any variation across 3 repeats (identical durations, intervals, and label distributions).
- **Across sessions**: small boundary effects — 15/79 matched files differed by >0.1s from the original pipeline (max diff = 0.59s, mean = 0.054s). This is consistent with CUDA autotuning / floating-point order-of-operations differences across different GPU nodes.

**Conclusion**: The model is deterministic within a run but has minor cross-session floating-point variation. This is NOT the cause of the large-scale failures (entire books with IoU ≈ 0). The between-run differences you noticed were likely from a different code path or threshold configuration.

## Threshold sweep results

Lowering the sigmoid threshold from the default 0.5 dramatically recovers "silent" files:

| Threshold | Files detected | Mean speech (s) |
|-----------|:--------------:|:---------------:|
| 0.10      | 95/96          | 23.5            |
| 0.20      | 93/96          | 17.1            |
| 0.30      | 88/96          | 13.6            |
| 0.40      | 77/96          | 11.2            |
| 0.50      | 67/96          | 9.3             |

### By category

| Category | At thresh=0.5 | At thresh=0.1 | Mean dur change |
|----------|:------------:|:------------:|:---------------:|
| vtc_silent (30 files) | 1/30 | 29/30 | 0.0s → 4.2s |
| vtc_under_detect (47 files) | 47/47 (4.6s avg) | 47/47 (29.9s avg) | +550% |
| good_baseline (18 files) | 18/18 (37.7s avg) | 18/18 (40.0s avg) | +6% (stable) |

Only 1 file (3067_LibriVox_en_seq_094) was truly undetectable at any threshold.

### Logit analysis

Raw logit inspection confirms the model IS seeing speech patterns — just below the 0.5 threshold:

- Many "silent" files have **MAL** activations at 0.30–0.49 across the majority of frames (e.g., `5882_LibriVox_en_seq_38`: MAL mean=0.32, 67/92 frames above 0.3)
- Some have **FEM** activations near threshold (e.g., `6158_LibriVox_en_seq_172`: FEM mean=0.33, 86/103 frames above 0.3)
- Good baseline files show near-maximum activations (0.95+), so threshold changes barely affect them (+6%)

The model trained on child-directed speech assigns partial confidence to adult audiobook narration — enough to detect it at a lower threshold, but not enough to breach 0.5 for any of its four classes (KCHI, OCH, MAL, FEM).

## Solution: Adaptive per-file thresholding

Since the VAD (TenVAD) reliably detects speech across all files, we use it as ground truth to calibrate VTC thresholds per file:

1. VTC inference runs once with `--save_logits`, producing raw logits for every file
2. A post-processing step (`scripts/adaptive_threshold.py`) loads VAD segments + VTC logits
3. For each file, it starts at threshold 0.5 and steps down until VTC–VAD IoU ≥ target (default 90%)
4. The highest threshold meeting the target is selected (preserves precision)
5. Final segments and per-file threshold metadata are saved

This avoids re-running GPU inference and handles the narrator-dependent sensitivity automatically.

## Recommended next steps

1. ~~**Listen** to samples from failing books~~ (placed in `data/vtc_samples/`) — confirmed real speech
2. ~~**Threshold sweep**~~ — completed, confirms threshold is the issue
3. ~~**Logit inspection**~~ — completed, model activates just below 0.5
4. **Run adaptive thresholding** — `scripts/adaptive_threshold.py` with `--target_iou 0.9`
5. **Decision** — evaluate whether adaptive thresholds produce acceptable labels, or if model fine-tuning on audiobook data is needed
