# VTC Test Suite

Tests for the `src/core/` modules. Covers pure interval arithmetic, VAD
processing helpers, activity-region logic, checkpoint I/O, metadata row
construction, parallel VAD execution, and end-to-end VAD on stitched long-form
audio fixtures.

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

Synthetic WAV files are committed under `tests/fixtures/` so the test suite
runs without access to external datasets:

| Directory | Contents |
|---|---|
| `tests/fixtures/good_book/` | 6 synthetic ~5 s chunks (sine + noise, 16 kHz mono) |
| `tests/fixtures/short_fail/` | 1 very short (~0.3 s) noise file |

The `stitched_audio_dir` session fixture in `conftest.py` concatenates these
chunks with configurable silence gaps to create long-form test recordings.

---

## Test files

### `test_intervals.py` — Pure interval arithmetic

Tests `src/core/intervals.py`, which is the mathematical foundation of
adaptive thresholding. Every threshold decision in the pipeline ultimately goes
through `compute_iou`, so correctness here is critical.

| Test class | What it verifies |
|---|---|
| `TestMergePairs` | Overlapping, touching, nested, unsorted, and duplicate `(onset, offset)` pairs all collapse to the correct non-overlapping result. |
| `TestTotalDuration` | Sum of interval widths, including empty and zero-width intervals. |
| `TestComputeIoU` | IoU between two interval sets. Covers both-empty → 1.0, one-empty → 0.0, partial overlap, full containment, and multi-segment cases. |
| `TestIntervalsToPairs` | Conversion from sample-index `(start_f, end_f, label)` triples to merged `(onset, offset)` pairs in seconds. |
| `TestIntervalsToSegments` | Conversion from sample-index triples to segment dicts with keys `uid, onset, offset, duration, label`. Verifies zero-duration segments are filtered out. |

---

### `test_regions.py` — Activity-region construction

Tests `src/core/regions.py`. Requires `torchcodec`/FFmpeg to import (skips
the whole module if unavailable at collection time). The tests themselves only
exercise the pure-Python `merge_into_activity_regions` and
`activity_region_coverage` functions — no model is loaded.

| Test class | What it verifies |
|---|---|
| `TestMergeIntoActivityRegions` | Padding is applied and clipped to `[0, file_duration]`. Segments within `merge_gap_s` collapse into one region; segments further apart remain separate. A dense sequence of small segments merges into a single region. |
| `TestActivityRegionCoverage` | Coverage fraction arithmetic: empty regions → 0.0, full file → 1.0, zero-duration file → 1.0 (guard against division by zero). |

---

### `test_vad_processing.py` — VAD helper functions

Tests `src/core/vad_processing.py`. The pure-logic tests (flag parsing,
segment conversion, resampling, error metadata) run on the login node. The
`process_vad_file` integration tests require TenVAD.

| Test class | What it verifies | TenVAD? |
|---|---|---|
| `TestGetRuns` | Flag array → `(speech_runs, silence_runs)` pairs. Checks all-speech, all-silence, alternating, single-frame, and precise boundary indexing. | No |
| `TestRunsToSegments` | Frame-index run arrays → `{onset, offset, duration}` dicts in seconds. Checks empty input and correct time arithmetic. | No |
| `TestSegmentStats` | Summary statistics (max/min/avg/sum/num) for a list of durations. Checks empty list and single/multiple values. | No |
| `TestResampleBlock` | Linear integer resampling at correct output length and `int16` dtype for upsample, downsample, and identity cases. | No |
| `TestVadErrorMetadata` | Error metadata rows have all required keys, `success=False`, and correct `file_id` derived from the path stem. | No |
| `TestProcessFile` | End-to-end: real audiobook files produce `success=True`, positive duration, at least one detected speech segment. A nonexistent file returns a clean error row without raising. | **Yes** |

---

### `test_checkpoint.py` — Resumable checkpoint I/O

Tests `src/core/checkpoint.py`. These protect the resume logic: an
interrupted run writes a checkpoint; on restart it is loaded and completed files
are skipped.

| Test class | What it verifies |
|---|---|
| `TestSaveCheckpoint` | Saving meta rows writes `_checkpoint_meta.parquet` with the correct content. Saving seg rows writes `_checkpoint_segs.parquet`. Empty rows produce no files. |
| `TestClearCheckpoint` | Both checkpoint files are removed after a successful run. Calling clear when no files exist does not raise. |

---

### `test_metadata.py` — VTC output row builders

Tests `src/core/metadata.py`. These validate the contract between the VTC
inference loop and the output parquet files.

| Test class | What it verifies |
|---|---|
| `TestVtcErrorRow` | Error rows contain all keys from `_EMPTY_VTC_META`, `vtc_status="error"`, and the correct `error` string. |
| `TestVtcMetaRow` | Label counts, total speech duration, and segment count are computed correctly from a list of segment dicts. Empty segments produce zero-filled rows. |
| `TestLoadVadReference` | Reads `vad_merged/*.parquet`, groups by `uid`, and returns merged `(onset, offset)` pairs. Returns an empty dict when the directory is missing. |

---

### `test_parallel.py` — Parallel VAD driver

Tests `src/core/parallel.py`. Requires TenVAD.

| Test | What it verifies |
|---|---|
| `test_single_file` | One file with one worker completes successfully and returns a metadata row with `success=True`. |
| `test_multiple_files_parallel` | Three files with two workers all complete. Tests the spawn-context process pool. |
| `test_with_checkpointing` | Passing a `checkpoint_dir` with `checkpoint_interval=1` doesn't crash and still returns correct results. |

---

### `test_stitched_audio.py` — Long-form audio fixtures + end-to-end integration

Tests `src/core/vad_processing.py` and `src/core/regions.py` together on
realistic long-form audio. Requires `torchcodec`/FFmpeg to import; TenVAD tests
within are additionally gated with `@requires_tenvad`.

**Fixtures** (`conftest.py`): Three audio files are synthesised once per test
session by stitching synthetic chunks from `tests/fixtures/` with silence:

| File | Construction | Speech ratio |
|---|---|---|
| `long_low_speech.wav` | 6 chunks + **60 s** silence between each | ~30 % |
| `long_high_speech.wav` | 6 chunks + **5 s** silence between each | ~50 % |
| `short_file.wav` | Single short chunk, copied as-is | — |

A `test_manifest.csv` containing paths to all three is also written.

| Test class | What it verifies | TenVAD? |
|---|---|---|
| `TestStitchedFixtures` | All three audio files and the manifest are created. Files are 16 kHz mono. The sparse file is longer than the dense file (more silence). | No |
| `TestVadOnStitchedAudio` | Sparse file has a lower VAD speech ratio than the dense file. Short file processes without error. | **Yes** |
| `TestActivityRegionsOnStitched` | After running real VAD, the sparse file produces activity regions covering less of the file than the dense file — validating the region-skipping optimisation threshold. The sparse file produces at least 2 distinct activity regions. | **Yes** |

---

## Shared infrastructure (`conftest.py`)

- **`stitched_audio_dir`** (`scope="session"`): builds the three test audio files
  once and returns the temp directory. All stitched-audio tests share this fixture
  without re-creating files.
- **`good_book_wavs`** / **`short_fail_wavs`** (`scope="session"`): sorted lists
  of WAV paths from `tests/fixtures/`. Tests skip if the fixture directory is
  empty.
- **`requires_tenvad`**: `pytest.mark.skipif` marker applied to any test or class
  that calls `process_vad_file` or `run_vad_parallel`. Skips with a human-readable
  reason rather than an import error.
