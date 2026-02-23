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

### Single-folder inference

Run the model on a folder of WAV files (16 kHz, mono):

```bash
uv run python -m src.pipeline.vtc my_data \
    --manifest my_audios/ \
    --device cuda
```

### Full pipeline on a SLURM cluster

Process an entire dataset end-to-end (VAD → VTC → comparison metrics):

```bash
# First run with a custom manifest:
bash slurm/pipeline.sh my_data \
    --manifest /data/recordings.xlsx \
    --path-col audio_path \
    --audio-root /store/audio/

# Subsequent runs (manifest already normalized):
bash slurm/pipeline.sh my_data
```

This submits three chained SLURM jobs with automatic dependency handling. See the [Pipeline](#3-pipeline) section for full documentation.

---

## 3. Pipeline

The pipeline runs three steps, each as a SLURM job:

| Step | Module | Resource | Description |
|------|--------|----------|-------------|
| **1. VAD** | `src.pipeline.vad` | 48 CPUs | TenVAD speech detection (multiprocessed, ~31 MB/worker) |
| **2. VTC** | `src.pipeline.vtc` | 4× GPU | VTC-2.0 inference with adaptive per-file thresholding |
| **3. Compare** | `src.pipeline.compare` | 8 CPUs | Per-file IoU / Precision / Recall + diagnostic figures |

**Adaptive thresholding:** The VTC model was trained on child-directed speech. For out-of-distribution audio (e.g., audiobook narrators), the default 0.5 sigmoid threshold can miss speech that the model does partially detect. Step 2 uses VAD output to automatically lower the threshold per file until VTC–VAD agreement reaches 90% IoU.

**Activity-region optimization:** For long recordings with sparse speech (< 90% coverage), VTC inference is restricted to speech-active regions only — cutting GPU time by up to 10×.

**Resume support:** Both VAD and VTC save checkpoints. Interrupted jobs can be resubmitted and will skip already-completed files.

---

## 4. Project Structure

```
VTC/
├── src/
│   ├── utils.py             # Shared utilities (manifest I/O, paths, logging)
│   ├── pipeline/            # CLI entry points (one per pipeline step)
│   │   ├── vad.py           #   Step 1: Voice activity detection
│   │   ├── vtc.py           #   Step 2: VTC inference
│   │   ├── compare.py       #   Step 3: VAD vs VTC comparison
│   │   ├── normalize.py     #   Manifest normalization
│   │   └── preflight.py     #   Pre-pipeline dataset scan
│   └── core/                # Reusable, tested modules
│       ├── intervals.py     #   Interval arithmetic (merge, IoU)
│       ├── regions.py       #   Activity-region optimization
│       ├── thresholds.py    #   Adaptive threshold sweeping
│       ├── vad_processing.py#   Per-file VAD (worker code)
│       ├── parallel.py      #   Process pool driver with progress queue
│       ├── checkpoint.py    #   Checkpoint save / resume
│       └── metadata.py      #   VTC metadata constructors
├── slurm/
│   ├── pipeline.sh          # One-command pipeline orchestrator
│   ├── vad.slurm            # SLURM: VAD (CPU, 48 workers)
│   ├── vtc.slurm            # SLURM: VTC (GPU array, 4 shards)
│   ├── compare.slurm        # SLURM: Compare (CPU)
│   └── test.slurm           # SLURM: pytest on compute node
├── tests/                   # pytest suite covering all core modules
│   ├── conftest.py          #   Stitched audio fixtures
│   ├── fixtures/            #   Synthetic WAV files (committed)
│   ├── test_intervals.py
│   ├── test_regions.py
│   ├── test_checkpoint.py
│   ├── test_metadata.py
│   ├── test_parallel.py
│   ├── test_vad_processing.py
│   └── test_stitched_audio.py
├── VTC-2.0/                 # Model weights & config
│   └── model/
│       ├── best.ckpt        #   Trained checkpoint (~1 GB, git-lfs)
│       └── config.yml       #   segma training config
├── manifests/               # Dataset manifests (one CSV per dataset)
├── output/                  # Pipeline outputs (per-dataset subdirs)
├── metadata/                # Per-file statistics (per-dataset subdirs)
├── figures/                 # Diagnostic plots (per-dataset subdirs)
├── logs/                    # SLURM logs + benchmark records
├── pyproject.toml           # Python project config (uv / pip)
├── requirements.txt         # Pinned dependency lockfile
└── check_sys_dependencies.sh
```

### Data flow

All paths are derived from the dataset name:

```
manifests/{dataset}.csv  →  output/{dataset}/   (segments, metrics, figures)
                         →  metadata/{dataset}/  (per-file statistics)
                         →  figures/{dataset}/   (diagnostic plots)
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