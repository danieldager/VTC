# DL++ Test Suite

Tests for the `src/core/`, `src/pipeline/`, and `src/packaging/` modules.
Covers pure interval arithmetic, VAD processing helpers, SNR extraction,
ESC, checkpoint I/O, metadata row construction, parallel VAD
execution, clip tiling, end-to-end VAD integration on real speech, and
reproducibility.

---

## Running the tests

### On the login node (fast, partial)

```bash
uv run python3 -m pytest tests/
```

Tests that require TenVAD are automatically skipped with a
clear message. Suitable for quick sanity checks after code changes.

### On a compute node via SLURM (full)

```bash
sbatch slurm/test.slurm
```

All tests run. Logs land in `logs/tests/pytest_<jobid>.out`. Runs on any
`erc-dupoux`, `gpu-p1`, or `gpu-p2` node (no GPU required — TenVAD is
CPU-only).

Pass extra pytest flags through with `"$@"`:

```bash
# Stop after first failure
sbatch slurm/test.slurm -x

# Run only a specific file
sbatch slurm/test.slurm tests/test_vad_processing.py
```

---

## Why some tests only run on compute nodes

TenVAD links against `libc++.so.1` from the LLVM runtime, which is not in the
login node's library path. The SLURM script loads `module load ffmpeg llvm` and
sets `LD_LIBRARY_PATH` accordingly. Tests depending on TenVAD are decorated with
`@requires_tenvad` and skip cleanly with an explanatory message rather than
erroring out when the library is unavailable.

---

## Test fixtures

Real speech WAV files are committed under `tests/fixtures/` so the test suite
runs without access to external datasets. These are short LibriVox excerpts
(public domain) chosen to cover the range of edge cases the pipeline handles:

| File | Duration | Content | Purpose |
|---|---|---|---|
| `speech_clean.wav` | 6.0s | Single male speaker, ~86% speech | Happy path — dense speech detection |
| `speech_multi.wav` | 10.2s | All 4 VTC labels (FEM+MAL+KCHI+OCH) | Multi-speaker / label diversity |
| `speech_sparse.wav` | 5.8s | Mostly silence, ~2.5% faint speech | Low speech ratio / sparse detection |
| `silence.wav` | 3.3s | Pure silence | Zero-speech edge case |
| `short.wav` | 0.3s | Very short clip | Minimum-length edge case |

Total: ~800 KB. See `tests/fixtures/LICENSE` for attribution.

---

## Test files

### `test_intervals.py` — Pure interval arithmetic

Tests `src/core/intervals.py`, which is the mathematical foundation of
adaptive thresholding.

| Test class | What it verifies |
|---|---|
| `TestMergePairs` | Overlapping, touching, nested, unsorted, and duplicate `(onset, offset)` pairs all collapse to the correct non-overlapping result. |
| `TestTotalDuration` | Sum of interval widths, including empty and zero-width intervals. |
| `TestComputeIoU` | IoU between two interval sets. Covers both-empty → 1.0, one-empty → 0.0, partial overlap, full containment, and multi-segment cases. |
| `TestIntervalsToPairs` | Conversion from sample-index `(start_f, end_f, label)` triples to merged `(onset, offset)` pairs in seconds. |
| `TestIntervalsToSegments` | Conversion from sample-index triples to segment dicts with keys `uid, onset, offset, duration, label`. Verifies zero-duration segments are filtered out. |

---

### `test_clips.py` — Clip tiling algorithm

Tests `src/packaging/clips.py`. Pure-Python — runs on the login node.

| Test class | What it verifies |
|---|---|
| `TestBuildActivityUnion` | Merging VTC + VAD segments into a non-overlapping activity mask. |
| `TestFindSilenceGaps` | Silence gap detection sorted longest-first. |
| `TestBuildClips` | Zero duration, short file → single clip, full coverage, segment assignment, duration sum = file duration. |
| `TestBuildClipsSplitting` | Splitting: prefers silence gaps, never cuts during activity, hard cut when no silence, even distribution, various fallback tiers. |
| `TestClipMetadata` | Relative timestamps, speech density, turns, labels, per-label properties. |

---

### `test_vad_processing.py` — VAD helper functions

Tests `src/core/vad_processing.py`. Pure-logic tests run on the login node.
Integration tests require TenVAD.

| Test class | What it verifies | TenVAD? |
|---|---|---|
| `TestGetRuns` | Flag array → run pairs. All-speech, all-silence, alternating, single-frame, boundaries. | No |
| `TestRunsToSegments` | Frame-index runs → time-domain segments. | No |
| `TestSegmentStats` | Summary statistics for duration lists. | No |
| `TestResampleBlock` | Linear integer resampling, correct dtype. | No |
| `TestVadErrorMetadata` | Error metadata has all required keys. | No |
| `TestProcessFile` | Real speech → detectable speech. Short file → no crash. Nonexistent → clean error. | **Yes** |

---

### `test_checkpoint.py` — Resumable checkpoint I/O

| Test class | What it verifies |
|---|---|
| `TestSaveCheckpoint` | Writes `_checkpoint_meta.parquet` and `_checkpoint_segs.parquet`. Empty rows → no files. |
| `TestClearCheckpoint` | Removes checkpoint files. Missing files don't raise. |

---

### `test_metadata.py` — VTC output row builders

| Test class | What it verifies |
|---|---|
| `TestVtcErrorRow` | Error rows have correct structure and `vtc_status="error"`. |
| `TestVtcMetaRow` | Label counts, speech duration, segment count computed correctly. |
| `TestLoadVadReference` | Reads `vad_merged/*.parquet` → `{uid: [(onset, offset), ...]}`. |

---

### `test_parallel.py` — Parallel VAD driver (TenVAD)

| Test | What it verifies |
|---|---|
| `test_single_file` | One file, one worker → `success=True`. |
| `test_multiple_files_parallel` | Three files, two workers → all succeed. |
| `test_with_checkpointing` | Checkpointing doesn't crash. |

---

### `test_stitched_audio.py` — End-to-end integration tests

Tests VAD and activity-region logic on real speech fixtures. Validates output
schemas and directional properties (speech files produce higher ratios than
silence files) without depending on specific numeric thresholds.

| Test class | What it verifies | TenVAD? |
|---|---|---|
| `TestFixtureFiles` | All fixtures exist, are 16 kHz mono, have expected durations. | No |
| `TestVadIntegration` | Speech → high ratio, silence → low ratio, short → no crash, ratio ordering, output schema, nonexistent file → clean error. | **Yes** |
| `TestActivityRegionsIntegration` | Speech → meaningful coverage, silence → low coverage, sparse → low coverage. | **Yes** |

---

### `test_snr.py` — Brouhaha SNR extraction

Tests `src/pipeline/snr.py`. Pure-numpy tests run on the login node.
GPU/model tests require Brouhaha (pyannote) and skip otherwise.

| Test class | What it verifies | Brouhaha? |
|---|---|---|
| `TestPoolSnr` | `pool_snr` aggregation: mean/median/percentile pooling, NaN handling, empty arrays, dtype. | No |
| `TestExtractSnr` | `_extract_snr` output schema, segment-level SNR values, short-file edge case. | **Yes** |
| `TestSnrRoundTrip` | End-to-end: WAV → extract → pool → stats are plausible. | **Yes** |
| `TestSnrReproducibility` | Bit-identical SNR values across reruns with fixed seeds. | **Yes** |

---

### `test_esc.py` — PANNs ESC

Tests `src/pipeline/esc.py`. Pure-numpy tests run on the login node.
GPU/model tests require PANNs CNN14 and skip otherwise.

| Test class | What it verifies | PANNs? |
|---|---|---|
| `TestMapToCategories` | `map_to_categories` mapping from AudioSet labels to ESC categories. | No |
| `TestPoolESC` | `pool_esc` aggregation over frame-level logits. | No |
| `TestPoolToCategories` | `pool_to_categories` top-k category extraction. | No |
| `TestClipESCIntegration` | End-to-end clip-level ESC on synthetic data. | No |
| `TestExtractPanns` | `extract_panns` output schema, real audio inference. | **Yes** |

---

### `test_reproducibility.py` — Deterministic pipeline outputs

Verifies bit-identical results across reruns using `set_seeds(42)`.

| Test class | What it verifies | Dependencies |
|---|---|---|
| `TestVadReproducibility` | Single file, all speech fixtures, silence, and short file all produce identical segments. | TenVAD |
| `TestVtcReproducibility` | Forward pass logits and thresholded segments are bit-identical. | TenVAD + segma model |

---

## Shared infrastructure (`conftest.py`)

- **Individual WAV fixtures** (`scope="session"`): `speech_clean_wav`,
  `speech_multi_wav`, `speech_sparse_wav`, `silence_wav`, `short_wav`.
- **Collection fixtures**: `all_fixture_wavs` (all 5), `speech_fixture_wavs`
  (3 with speech), `good_book_wavs` (backward-compat alias).
- **`test_manifest`**: writes a manifest CSV for all fixture WAVs.
- **`requires_tenvad`** / **`requires_torchcodec`** / **`requires_brouhaha`**:
  skip markers with human-readable reasons.
