# Voice Type Classifier (VTC) 2.0

A speaker segmentation model that classifies speech into four speaker types (**FEM**, **MAL**, **KCHI**, **OCH**) in child-centered long-form audio recordings. Includes a production-ready SLURM pipeline for processing datasets of any size — from a handful of files to hundreds of thousands.

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Pipeline](#3-pipeline)
4. [Project Structure](#4-project-structure)
5. [Model Performance](#5-model-performance)
6. [Citation](#6-citation)
7. [Acknowledgement](#7-acknowledgement)

---

## 1. Installation

**Requirements:** Linux or macOS, Python ≥ 3.13, [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), [git-lfs](https://git-lfs.com/).

```bash
# Check system dependencies:
./check_sys_dependencies.sh

# Clone (includes model weights via git-lfs):
git lfs install
git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git
cd VTC

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

### Single-folder inference

Run the VTC model on a folder of audio files:

```bash
uv run python -m src.pipeline.vtc my_data \
    --manifest manifests/my_dataset.csv \
```

### Full pipeline on a SLURM cluster

Process an entire dataset end-to-end (VAD → VTC → comparison metrics):

```bash
# First run with a custom manifest:
bash slurm/pipeline.sh my_data \
    --manifest /data/recordings.parquet \
    --path-col audio_path \
    --audio-root /store/audio/

# Subsequent runs (manifest already normalized):
bash slurm/pipeline.sh my_data
```

This submits three chained SLURM jobs with automatic dependency handling. See the [Pipeline](#3-pipeline) section for full documentation.

---

## 3. Pipeline

The pipeline runs four steps, each as a SLURM job:

| Step | Module | Resource | Description |
|------|--------|----------|-------------|
| **1. VAD** | `src.pipeline.vad` | CPU | TenVAD speech detection (multiprocessed, ~31 MB/worker) |
| **2. VTC** | `src.pipeline.vtc` | GPU | VTC-2.0 inference with default threshold (0.5) |
| **3. SNR** | `src.pipeline.snr` | GPU | Brouhaha SNR/C50 extraction per audio file |
| **4. Noise** | `src.pipeline.noise` | GPU | PANNs CNN14 noise classification per audio file |
| **5. Package** | `src.pipeline.package` | CPU | ≤10-min clipping → WebDataset shards + plots |

**Resume support:** Both VAD and VTC save checkpoints. Interrupted jobs can be resubmitted and will skip already-completed files.

### Packaging (Step 5)

The packaging step tiles full audio files into clips of roughly equal length, cutting only at silence gaps (never mid-speech). Cut-point selection uses a **6-tier fallback chain**:

| Tier | Strategy | Severity |
|------|----------|----------|
| 1 | Long silence gap (≥10 s) in VAD∪VTC union | Clean |
| 2 | Any silence gap in VAD∪VTC union | Clean |
| 3 | Gap in VAD-only mask (VTC still active) | Info |
| 4 | Gap in VTC-only mask (VAD still active) | Info |
| 5 | VTC speaker-change boundary (inside active audio) | Warning |
| 6 | Hard cut — no gaps or boundaries | Warning |

Within each tier, the midpoint closest to the ideal evenly-distributed position is chosen. The pipeline output includes a **tier breakdown** showing how many cuts used each strategy.

Output:
- `output/{dataset}/shards/` — WebDataset `.tar` shards (WAV/FLAC + JSON metadata)
- `output/{dataset}/shards/manifest.csv` — per-clip metadata
- `output/{dataset}/shards/samples/` — random sample clips for manual validation

### Clip metadata

Each clip in a shard is stored as up to four files sharing the key `{uid}_{clip_idx:04d}`:

| File | Format | Contents |
|------|--------|----------|
| `{clip_id}.flac` / `.wav` | FLAC / WAV | Mono audio, 16 kHz |
| `{clip_id}.json` | JSON (UTF-8) | All scalar + structured metadata (see below) |
| `{clip_id}.snr.npy` | NumPy float16 | Time-series SNR values per `snr_step_s` window *(optional)* |
| `{clip_id}.c50.npy` | NumPy float16 | Time-series C50 values per `snr_step_s` window *(optional)* |

The `.json` sidecar contains:

**Provenance** — `uid`, `clip_idx`, `clip_id`, `abs_onset`, `abs_offset`, `duration` (seconds within the source file), `source_path`, `audio_fmt`, `sample_rate`.

**VTC speech** — `vtc_speech_duration`, `vtc_speech_density`, `n_vtc_segments`, `mean_vtc_seg_duration`, `mean_vtc_gap`, `n_turns`, `n_labels`, `labels_present` (list of speaker types), `has_adult`, `dominant_label`, `label_durations` (seconds per type), `vad_coverage_by_label` (fraction of each VTC label also covered by VAD).

**Demographics** — `child_speech_duration`, `adult_speech_duration`, `child_fraction` (share of VTC speech that is child).

**VAD speech** — `vad_speech_duration`, `vad_speech_density`, `n_vad_segments`.

**VAD–VTC agreement** — `vad_vtc_iou`: frame-level Intersection over Union between the two systems' masks.

**SNR** *(Brouhaha; null if not run)* — `snr_mean`, `snr_std`, `snr_min`, `snr_max` (dB); `snr_step_s` (window size); `snr` (full time-series list, also stored as `.snr.npy`).

**C50 clarity** *(Brouhaha; null if not run)* — `c50_mean`, `c50_std`, `c50_min`, `c50_max` (dB); `c50` (full time-series, also stored as `.c50.npy`). Higher C50 = less reverberation.

**Noise environment** *(PANNs; null if not run)* — `dominant_noise` (category name), `noise_profile` (dict of mean probability per category).

**Segment detail** — `vad_segments` and `vtc_segments`: lists of `{onset, offset, duration}` objects with timestamps relative to the clip start. `vtc_segments` additionally carry a `label` field (FEM / MAL / KCHI / OCH).

---

## 4. Project Structure

```
VTC/
├── src/
│   ├── utils.py             # Shared utilities (manifest I/O, paths, logging)
│   ├── compat.py            # Compatibility shims (torchaudio patches)
│   ├── pipeline/            # CLI entry points (one per pipeline step)
│   │   ├── vad.py           #   Step 1: Voice activity detection
│   │   ├── vtc.py           #   Step 2: VTC inference (diarization)
│   │   ├── snr.py           #   Step 3: Brouhaha SNR/C50 extraction
│   │   ├── noise.py         #   Step 4: PANNs CNN14 noise classification
│   │   ├── package.py       #   Step 5: Audio clipping + WebDataset shards
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
│       ├── dashboard.py     #   Orchestrator (calls sub-modules)
│       ├── dashboard_snr.py #   SNR quality + noise environment pages
│       ├── dashboard_speech.py # Conversation + turns pages
│       ├── dashboard_overview.py # Overview + correlation + text summary
│       └── packaging.py     #   Per-clip/label summary grids
├── slurm/
│   ├── pipeline.sh          # One-command pipeline orchestrator
│   ├── vad.slurm            # SLURM: VAD (CPU, 48 workers)
│   ├── vtc.slurm            # SLURM: VTC (GPU array, 4 shards)
│   ├── snr.slurm            # SLURM: Brouhaha SNR (GPU)
│   ├── noise.slurm          # SLURM: PANNs noise (GPU)
│   ├── package_test.sh      # SLURM: End-to-end packaging test
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
├── scripts/
│   ├── download_brouhaha.py # Auto-download Brouhaha checkpoint
│   └── make_manifest.py     # Generate manifest from audio directory
├── models/                  # Brouhaha checkpoint (gitignored, auto-downloaded)
│   └── best/checkpoints/
│       └── best.ckpt        #   ~47 MB, from ylacombe/brouhaha-best
├── VTC-2.0/                 # Model weights & config
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

## 5. Model Performance

### Speaker classes

| Class | Description |
|-------|-------------|
| **KCHI** | Key child speech |
| **OCH** | Other child speech |
| **MAL** | Adult male speech |
| **FEM** | Adult female speech |

The model is trained on child-centered long-form recordings collected using a portable recorder attached to a child's vest (typically 0–5 years old).

### F1 scores on the held-out test set

| Model | KCHI | OCH | MAL | FEM | Average |
|-------|:----:|:---:|:---:|:---:|:-------:|
| VTC 1.0 | 68.2 | 30.5 | 41.2 | 63.7 | 50.9 |
| VTC 1.5 | 68.4 | 20.6 | 56.7 | 68.9 | 53.6 |
| **VTC 2.0** | **71.8** | **51.4** | **60.3** | **74.8** | **64.6** |
| Human 2 | 79.7 | 60.4 | 67.6 | 71.5 | 69.8 |

VTC 2.0 surpasses human-level performance on the **FEM** class.

### Runtime

<table>
<tr><th>GPU</th><th>CPU</th></tr>
<tr><td>

| Batch size | Hardware | Speedup |
|:----------:|:---------|:-------:|
| 256 | H100 | **1/905** |
| 256 | A40 | 1/650 |
| 256 | Quadro RTX 8000 | 1/531 |

</td><td>

| Batch size | Hardware | Speedup |
|:----------:|:---------|:-------:|
| 256 | AMD EPYC 9334 | **1/29** |
| 256 | AMD EPYC 7453 | 1/22 |
| 256 | Xeon Silver 4214R | 1/16 |

</td></tr>
</table>

Speedup factor is relative to audio duration. For example, 1/905 means 1 hour of audio processes in ~4 seconds on an H100.

### Confusion matrices

<p float="left" align="middle">
  <img src="figures/vtc2_heldout_full_cm_precision.png" width="400"/>
  <img src="figures/vtc2_heldout_full_cm_recall.png" width="400"/>
</p>

---

## 6. Citation

The training code for BabyHuBERT can be found at [LAAC-LSCP/BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT).

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

---

## 7. Acknowledgement

The Voice Type Classifier has benefited from numerous contributions over time.

### VTC 1.5 (Whisper-VTC)

GitHub: [LAAC-LSCP/VTC-IS-25](https://github.com/LAAC-LSCP/VTC-IS-25)

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

### VTC 1.0 (PyanNet-VTC)

GitHub: [MarvinLvn/voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier)

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

This work uses the [segma](https://github.com/arxaqapi/segma) library, inspired by [pyannote.audio](https://github.com/pyannote/pyannote-audio).

This work was performed using HPC resources from GENCI-IDRIS (Grant 2024-AD011015450 and 2025-AD011016414) and was developed as part of the ExELang project funded by the European Union (ERC, ExELang, Grant No 101001095).