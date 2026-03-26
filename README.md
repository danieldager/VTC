# DL++ (Dataloader++)

A feature processing and data loading framework for child-centered long-form audio recordings. Runs a SLURM pipeline that extracts speech activity, speaker types, signal quality, and environmental noise — then packages everything into WebDataset shards with rich per-clip metadata for model training.

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Pipeline](#3-pipeline)
4. [Project Structure](#4-project-structure)
5. [Dataloader](#5-dataloader)
6. [Citation](#6-citation)
7. [Component Models](#7-component-models)
8. [Acknowledgements](#8-acknowledgements)

---

## 1. Installation

**Requirements:** Linux or macOS, Python ≥ 3.13, [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), [git-lfs](https://git-lfs.com/).

```bash
# Check system dependencies:
./check_sys_dependencies.sh

# Clone (includes model weights via git-lfs):
git lfs install
git clone --recurse-submodules https://github.com/LAAC-LSCP/DLplusplus.git
cd DLplusplus

# Install Python dependencies:
uv sync

# Download the Brouhaha SNR model checkpoint (~47 MB, one-time):
uv run python scripts/download_brouhaha.py
```

<details>
<summary>Alternative: pip install</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

</details>

---

## 2. Quick Start

### Generate a manifest from a directory

If your audio files live in a directory (no pre-existing metadata file), generate
a ready-to-use manifest with:

```bash
python scripts/make_manifest.py /path/to/audio/ -name my_dataset
```

This recursively scans for all common audio formats (`wav`, `flac`, `mp3`, `ogg`,
`opus`, `m4a`, `aac`, `aiff`, `wma`) and writes `manifests/my_dataset.csv` with
columns `path` (absolute), `uid` (filename stem), and `ext` (format).

### Single-step inference

Run an individual pipeline step on a folder of audio files:

```bash
# Speaker diarization (VTC)
uv run python -m src.pipeline.vtc my_data \
    --manifest manifests/my_dataset.csv

# Voice activity detection (VAD)
uv run python -m src.pipeline.vad my_data \
    --manifest manifests/my_dataset.csv
```

### Full pipeline on a SLURM cluster

Process an entire dataset end-to-end:

```bash
# First run with a custom manifest:
bash slurm/pipeline.sh my_data \
    --manifest manifests/my_dataset.parquet \
    --path-col audio_path \
    --audio-root /store/audio/

# Subsequent runs (manifest already normalized):
bash slurm/pipeline.sh my_data
```

This submits five SLURM jobs — four feature extraction steps in parallel, then a packaging step that depends on all four. See the [Pipeline](#3-pipeline) section for full documentation.

---

## 3. Pipeline

The pipeline orchestrator (`slurm/pipeline.sh`) runs a preflight check, then submits five SLURM jobs:

```
                ┌─── VAD  (CPU)  ───┐
                ├─── VTC  (GPU)  ───┤
Raw Audio ──►   ├─── SNR  (GPU)  ───┼──► Package (CPU)
                └─── Noise (GPU) ───┘
```

Steps 1–4 run **in parallel** as independent jobs. Step 5 (Package) depends on all four completing successfully.

| Step | Module | Resource | Description |
|------|--------|----------|-------------|
| **1. VAD** | `src.pipeline.vad` | CPU | TenVAD speech activity detection |
| **2. VTC** | `src.pipeline.vtc` | GPU | BabyHuBERT/segma speaker diarization (KCHI, OCH, MAL, FEM) |
| **3. SNR** | `src.pipeline.snr` | GPU | Brouhaha per-frame SNR & C50 extraction |
| **4. Noise** | `src.pipeline.noise` | GPU | PANNs CNN14 environmental noise classification |
| **5. Package** | `src.pipeline.package` | CPU | Clip tiling + WebDataset shards + dashboards |

**Resume support:** VAD and VTC save checkpoints. Interrupted jobs can be resubmitted and will skip already-completed files.

### Step 1 — VAD (Voice Activity Detection)

Runs [TenVAD](https://github.com/Tencent/TenVAD) with CPU multiprocessing (default: all cores).

**Output:**
- `output/{dataset}/vad_raw/segments.parquet` — per-frame VAD segments
- `output/{dataset}/vad_merged/segments.parquet` — merged overlapping segments
- `output/{dataset}/vad_meta/metadata.parquet` — per-file summary metadata

### Step 2 — VTC (Voice Type Classification)

Runs the BabyHuBERT model via [segma](https://github.com/arxaqapi/segma) on GPU (SLURM array, default 3 shards).

**Output:**
- `output/{dataset}/vtc_raw/` — raw VTC segments (per-shard parquets)
- `output/{dataset}/vtc_merged/` — merged/deduplicated segments across shards
- `output/{dataset}/vtc_meta/` — per-file summary metadata

Segment columns: `uid`, `onset`, `offset`, `duration`, `label` (FEM / MAL / KCHI / OCH).

### Step 3 — SNR (Signal-to-Noise Ratio & Clarity)

Runs [Brouhaha](https://github.com/marianne-m/brouhaha-vad) on GPU (SLURM array, default 2 shards). Produces **per-file time-series arrays** and **speech-masked summary statistics**.

**Output:**
- `output/{dataset}/snr/{uid}.npz` — per-file compressed arrays:
  - `snr` (float16, shape `n_frames`) — per-frame SNR in dB
  - `c50` (float16, shape `n_frames`) — per-frame C50 clarity in dB
  - `vad` (float16, shape `n_frames`) — per-frame Brouhaha VAD probability
  - `step_s` — frame step in seconds (~16 ms)
  - `vad_threshold` — threshold used (0.5)
- `output/{dataset}/snr_meta/shard_{id}.parquet` — per-file metadata:
  - `uid`, `snr_status`, `duration`, `n_raw_frames`, `n_speech_frames`, `speech_fraction`
  - `snr_mean`, `snr_std`, `snr_min`, `snr_max` — computed only on speech frames (VAD > 0.5)
  - `c50_mean`, `c50_std`, `c50_min`, `c50_max` — computed only on speech frames

Downstream steps (e.g. packaging) index into the per-frame arrays by `onset/offset` using `step_s` to compute exact segment-level statistics.

### Step 4 — Noise (Environmental Sound Classification)

Runs [PANNs CNN14](https://github.com/qiuqiangkong/panns_inference) on GPU (SLURM array, default 2 shards). Classifies audio into 13 coarse categories and 527 AudioSet classes.

**Output:**
- `output/{dataset}/noise/{uid}.npz` — per-file compressed arrays:
  - `categories` (float16, shape `n_bins × 13`) — coarse category probabilities
  - `category_names` — the 13 category labels
  - `audioset_probs` (float16, shape `n_bins × 527`) — full AudioSet probabilities
  - `audioset_names` — 527 AudioSet display labels
  - `pool_step_s`, `inference_step_s` — time resolutions
- `output/{dataset}/noise_meta/shard_{id}.parquet` — per-file metadata:
  - `uid`, `noise_status`, `duration`, `n_inference_windows`, `n_pooled_bins`
  - `dominant_category`, `dominant_prob`
  - `prob_{category}` — mean probability for each of 13 categories

**Categories:** alarm_signal, animal, crying, environment, human_activity, impact, laughter, machinery, music, nature, other, silence, singing, tv_radio, vehicle.

### Step 5 — Package (Clip Tiling + WebDataset Shards)

Tiles full audio files into clips of roughly equal length, cutting only at silence gaps (never mid-speech). Cut-point selection uses a **6-tier fallback chain**:

| Tier | Strategy | Severity |
|------|----------|----------|
| 1 | Long silence gap (≥10 s) in VAD∪VTC union | Clean |
| 2 | Any silence gap in VAD∪VTC union | Clean |
| 3 | Gap in VAD-only mask (VTC still active) | Info |
| 4 | Gap in VTC-only mask (VAD still active) | Info |
| 5 | VTC speaker-change boundary (inside active audio) | Warning |
| 6 | Hard cut — no gaps or boundaries | Warning |

Within each tier, the midpoint closest to the ideal evenly-distributed position is chosen. The pipeline output includes a **tier breakdown** showing how many cuts used each strategy.

**Output:**
- `output/{dataset}/shards/` — WebDataset `.tar` shards (WAV/FLAC + JSON metadata)
- `output/{dataset}/shards/manifest.csv` — per-clip metadata
- `output/{dataset}/shards/samples/` — random sample clips for manual validation
- `output/{dataset}/stats/` — Parquet DataFrames at multiple granularities (clip, segment, turn, conversation, file)
- `figures/{dataset}/dashboard/` — 6 PNG diagnostic dashboards (see `src/plotting/README.md`)

### Clip metadata

Each clip in a shard is stored as up to four files sharing the key `{uid}_{clip_idx:04d}`:

| File | Format | Contents |
|------|--------|----------|
| `{clip_id}.wav` / `.flac` | WAV / FLAC | Mono audio, 16 kHz |
| `{clip_id}.json` | JSON (UTF-8) | All scalar + structured metadata (see below) |
| `{clip_id}.snr.npy` | NumPy float16 | Time-series SNR values per `snr_step_s` window |
| `{clip_id}.c50.npy` | NumPy float16 | Time-series C50 values per `snr_step_s` window |

The `.json` metadata contains:

**Source** — `uid`, `clip_idx`, `clip_id`, `abs_onset`, `abs_offset`, `duration`, `source_path`, `audio_fmt`, `sample_rate`.

**VTC speech** — `vtc_speech_duration`, `vtc_speech_density`, `n_vtc_segments`, `mean_vtc_seg_duration`, `mean_vtc_gap`, `n_turns`, `n_labels`, `labels_present`, `has_adult`, `dominant_label`, `label_durations`, `vad_coverage_by_label` (fraction of each VTC label also covered by VAD).

**Demographics** — `child_speech_duration`, `adult_speech_duration`, `child_fraction` (share of VTC speech that is child).

**VAD speech** — `vad_speech_duration`, `vad_speech_density`, `n_vad_segments`.

**VAD–VTC agreement** — `vad_vtc_iou`: frame-level Intersection over Union between the two systems' masks.

**SNR** — `snr_mean`, `snr_std`, `snr_min`, `snr_max` (dB); `snr_step_s` (window size); `snr` (full time-series list, also stored as `.snr.npy`).

**C50** — `c50_mean`, `c50_std`, `c50_min`, `c50_max` (dB); `c50` (full time-series, also stored as `.c50.npy`). Higher C50 = less reverberation.

**Noise environment** — `dominant_noise` (category name), `noise_profile` (dict of mean probability per category).

**Segment detail** — `vad_segments` and `vtc_segments`: lists of `{onset, offset, duration}` objects with timestamps relative to the clip start. `vtc_segments` additionally carry a `label` field (FEM / MAL / KCHI / OCH).

### Additional tools

| Module | Purpose |
|--------|---------|
| `src.pipeline.compare` | VAD vs VTC comparison (IoU, precision, recall, diagnostics) |
| `src.pipeline.normalize` | Standardize external manifests into `manifests/{dataset}.csv` |
| `src.pipeline.preflight` | Estimate dataset size, GPU needs, and wall-clock time |
| `src.pipeline.segment_snr` | Post-hoc per-VTC-segment SNR/C50 averaging |

---

## 4. Project Structure

```
DLplusplus/
├── src/
│   ├── utils.py             # Shared utilities (manifest I/O, paths, logging)
│   ├── compat.py            # Compatibility shims (torchaudio patches)
│   ├── pipeline/            # CLI entry points (one per pipeline step)
│   │   ├── vad.py           #   Step 1: TenVAD voice activity detection
│   │   ├── vtc.py           #   Step 2: BabyHuBERT speaker diarization
│   │   ├── snr.py           #   Step 3: Brouhaha SNR/C50 extraction
│   │   ├── noise.py         #   Step 4: PANNs CNN14 noise classification
│   │   ├── package.py       #   Step 5: Audio clipping + WebDataset shards
│   │   ├── segment_snr.py   #   Post-hoc per-segment SNR/C50 averaging
│   │   ├── compare.py       #   VAD vs VTC comparison helpers
│   │   ├── normalize.py     #   Manifest normalization
│   │   └── preflight.py     #   Pre-pipeline dataset scan
│   ├── packaging/           # Clip building, shard writing, listener
│   │   ├── clips.py         #   Clip tiling algorithm (6-tier fallback)
│   │   ├── stats.py         #   Per-clip/file/conversation statistics
│   │   ├── writer.py        #   WebDataset tar shard writer
│   │   └── listener.py      #   Sample extraction for validation
│   ├── core/                # Reusable, tested modules
│   │   ├── intervals.py     #   Interval arithmetic (merge, IoU)
│   │   ├── conversations.py #   Turn/conversation extraction
│   │   ├── vad_processing.py#   Per-file VAD (worker code)
│   │   ├── parallel.py      #   Process pool driver with progress queue
│   │   ├── checkpoint.py    #   Checkpoint save / resume
│   │   └── metadata.py      #   VTC metadata constructors
│   └── plotting/            # Dashboard figure generation
│       ├── figures.py       #   Orchestrator (calls sub-modules)
│       ├── snr_noise.py     #   SNR quality + noise environment
│       ├── speech_turns.py  #   Conversational structure + turns
│       ├── overview.py      #   Dataset overview + correlation + text summary
│       └── packaging.py     #   Per-clip/label summary grids
├── dataloader/              # Dataloader++ package (see Section 5)
│   ├── types.py             #   Shared type aliases and enums
│   ├── processor/           #   Feature Processor ABCs (offline extraction)
│   │   ├── base.py          #     FeatureProcessor ABC
│   │   └── registry.py      #     Processor discovery & registration
│   ├── loader/              #   Feature Loader ABCs (waveform + metadata I/O)
│   │   ├── base.py          #     FeatureLoader ABC
│   │   ├── waveform.py      #     WaveformLoader
│   │   └── metadata.py      #     MetadataLoader (JSON/Parquet/NPZ)
│   ├── manifest/            #   Manifest management
│   │   ├── schema.py        #     MetadataManifest schema
│   │   ├── joiner.py        #     ManifestJoiner (Big Join)
│   │   └── store.py         #     MetadataStore (unified I/O)
│   ├── transform/           #   Runtime data transforms
│   │   ├── base.py          #     DataProcessor ABC + Compose
│   │   ├── audio.py         #     Resample, segment, normalize
│   │   └── label.py         #     Label encoding, mask generation
│   ├── batch/               #   Batching and collation
│   │   ├── base.py          #     Collator ABC
│   │   ├── data_batch.py    #     DataBatch container
│   │   └── speech.py        #     SpeechCollator implementation
│   └── dataset/             #   PyTorch Dataset implementations
│       ├── base.py          #     SpeechDataset ABC
│       └── webdataset.py    #     WebDataset-backed loader
├── slurm/
│   ├── pipeline.sh          # One-command pipeline orchestrator
│   ├── vad.slurm            # SLURM: VAD (CPU, 48 workers)
│   ├── vtc.slurm            # SLURM: VTC (GPU array, 3 shards)
│   ├── snr.slurm            # SLURM: Brouhaha SNR (GPU array, 2 shards)
│   ├── noise.slurm          # SLURM: PANNs noise (GPU array, 2 shards)
│   ├── segment_snr.slurm    # SLURM: Per-segment SNR (GPU array)
│   ├── vtc_clips.slurm      # SLURM: VTC on packaged clips
│   ├── snr_diagnostic.slurm # SLURM: SNR masking diagnostics
│   ├── package_test.sh      # Quick end-to-end packaging test
│   ├── repackage_test.sh    # Re-package + clip alignment test
│   └── test.slurm           # SLURM: pytest on compute node
├── tests/                   # pytest suite covering all core modules
│   ├── conftest.py          #   Audio fixtures + skip markers
│   ├── fixtures/            #   Short WAV files (committed)
│   ├── test_intervals.py
│   ├── test_checkpoint.py
│   ├── test_metadata.py
│   ├── test_parallel.py
│   ├── test_clips.py        #   Clip tiling + tier fallback chain
│   ├── test_snr.py          #   Brouhaha SNR extraction
│   ├── test_noise.py        #   PANNs noise classification
│   ├── test_vad_processing.py
│   ├── test_reproducibility.py
│   └── test_stitched_audio.py
├── docs/
│   └── DATALOADER_DESIGN.md # Dataloader++ specification
├── scripts/
│   ├── download_brouhaha.py # Auto-download Brouhaha checkpoint
│   └── make_manifest.py     # Generate manifest from audio directory
├── models/                  # Brouhaha checkpoint (gitignored, auto-downloaded)
│   └── best/checkpoints/
│       └── best.ckpt        #   ~47 MB, from ylacombe/brouhaha-best
├── VTC-2.0/                 # BabyHuBERT model weights & config
│   └── model/
│       ├── best.ckpt        #   Trained checkpoint (~1 GB, git-lfs)
│       └── config.yml       #   segma training config
├── manifests/               # Dataset manifests (one CSV per dataset)
├── output/                  # Pipeline outputs (per-dataset subdirs)
├── figures/                 # Diagnostic plots (per-dataset subdirs)
├── logs/                    # SLURM logs + benchmark records
├── pyproject.toml           # Python project config (uv / pip)
├── requirements.txt         # Pinned dependency lockfile
└── check_sys_dependencies.sh
```

### Data flow

All paths are derived from the dataset name:

```
manifests/{dataset}.csv  →  output/{dataset}/   (metadata, segments, metrics)
                         →  figures/{dataset}/  (plots)
```

### Running tests

```bash
# Login node (TenVAD tests auto-skip):
uv run python3 -m pytest tests/

# Compute node (full suite):
sbatch slurm/test.slurm
```

---

## 5. Dataloader

The `dataloader/` package implements the **Dataloader++** specification for Meta's speech training infrastructure. It bridges the offline feature processing pipeline (above) with online model training.

See [`docs/DATALOADER_DESIGN.md`](docs/DATALOADER_DESIGN.md) for the full design document.

| Component | Location | Purpose |
|-----------|----------|---------|
| **Feature Processor** | `dataloader/processor/` | ABC wrapping offline extraction stages (VAD, VTC, SNR, Noise) |
| **Feature Loader** | `dataloader/loader/` | Load waveforms + metadata from WebDataset shards or raw files |
| **Manifest Joiner** | `dataloader/manifest/` | Join heterogeneous metadata manifests by `wav_id` (the "Big Join") |
| **Data Processor** | `dataloader/transform/` | Composable runtime transforms (segment, resample, encode, mask) |
| **Collator / DataBatch** | `dataloader/batch/` | Pad variable-length samples into typed `DataBatch` tensors |
| **Dataset** | `dataloader/dataset/` | PyTorch `Dataset` implementations (WebDataset-backed) |

---

## 6. Citation

```bibtex
@software{dlplusplus,
    title  = {{DL++}: Feature Processing and Data Loading for Child-Centered Long-Form Audio},
    author = {Dager, Daniel and Kunze, Tarek and Charlot, Théo and Cristia, Alejandrina and Dupoux, Emmanuel and Lavechin, Marvin},
    year   = {2026},
    url    = {https://github.com/LAAC-LSCP/DLplusplus},
}
```

---

## 7. Component Models

DL++ integrates the following models as feature processing stages:

### TenVAD — Voice Activity Detection

[Tencent/TenVAD](https://github.com/Tencent/TenVAD) — lightweight speech activity detector used in Step 1 (CPU).

### BabyHuBERT — Voice Type Classification (VTC 2.0)

Speaker diarization into four types (KCHI, OCH, MAL, FEM), trained on child-centered long-form recordings. Used in Step 2 (GPU).

Training code: [LAAC-LSCP/BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT)

```bibtex
@misc{charlot2025babyhubertmultilingualselfsupervisedlearning,
    title={BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings},
    author={Théo Charlot and Tarek Kunze and Maxime Poli and Alejandrina Cristia and Emmanuel Dupoux and Marvin Lavechin},
    year={2025},
    eprint={2509.15001},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2509.15001},
}
```

<details>
<summary>Earlier VTC versions</summary>

**VTC 1.5 (Whisper-VTC)** — GitHub: [LAAC-LSCP/VTC-IS-25](https://github.com/LAAC-LSCP/VTC-IS-25)

```bibtex
@inproceedings{kunze25_interspeech,
    title     = {{Challenges in Automated Processing of Speech from Child Wearables: The Case of Voice Type Classifier}},
    author    = {Tarek Kunze and Marianne Métais and Hadrien Titeux and Lucas Elbert and Joseph Coffey and Emmanuel Dupoux and Alejandrina Cristia and Marvin Lavechin},
    year      = {2025},
    booktitle = {{Interspeech 2025}},
    pages     = {2845--2849},
    doi       = {10.21437/Interspeech.2025-1962},
}
```

**VTC 1.0 (PyanNet-VTC)** — GitHub: [MarvinLvn/voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier)

```bibtex
@inproceedings{lavechin20_interspeech,
    title     = {An Open-Source Voice Type Classifier for Child-Centered Daylong Recordings},
    author    = {Marvin Lavechin and Ruben Bousbib and Hervé Bredin and Emmanuel Dupoux and Alejandrina Cristia},
    year      = {2020},
    booktitle = {Interspeech 2020},
    pages     = {3072--3076},
    doi       = {10.21437/Interspeech.2020-1690},
}
```

</details>

### Brouhaha — SNR & C50 Estimation

[marianne-m/brouhaha-vad](https://github.com/marianne-m/brouhaha-vad) — per-frame signal-to-noise ratio and clarity (C50) extraction. Used in Step 3 (GPU).

```bibtex
@inproceedings{lavechin2023brouhaha,
    title     = {Brouhaha: Multi-task Training for Voice Activity Detection, Speech-to-Noise Ratio, and Speech Reverberation Estimation},
    author    = {Marvin Lavechin and Marianne Métais and Hadrien Titeux and Alodie Boissonnet and Johan Music and Hervé Bredin and Emmanouil Benetos and Alejandrina Cristia},
    year      = {2023},
    booktitle = {2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
    doi       = {10.1109/ASRU57964.2023.10389642},
}
```

### PANNs CNN14 — Environmental Noise Classification

[qiuqiangkong/panns_inference](https://github.com/qiuqiangkong/panns_inference) — AudioSet-based sound event detection (527 classes, grouped into 13 coarse categories). Used in Step 4 (GPU).

```bibtex
@inproceedings{kong2020panns,
    title     = {PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition},
    author    = {Qiuqiang Kong and Yin Cao and Turab Iqbal and Yuxuan Wang and Wenwu Wang and Mark D. Plumbley},
    year      = {2020},
    journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    volume    = {28},
    pages     = {2880--2894},
    doi       = {10.1109/TASLP.2020.3030497},
}
```

---

## 8. Acknowledgements

This work uses the [segma](https://github.com/arxaqapi/segma) library, inspired by [pyannote.audio](https://github.com/pyannote/pyannote-audio).

This work was performed using HPC resources from GENCI-IDRIS (Grant 2024-AD011015450 and 2025-AD011016414) and was developed as part of the ExELang project funded by the European Union (ERC, ExELang, Grant No 101001095).