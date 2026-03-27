# DL++ Feature Extraction & Dataloading — Technical Summary

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Feature Extraction Stages](#2-feature-extraction-stages)
   - [Stage 1: VAD (Voice Activity Detection)](#stage-1-vad--voice-activity-detection)
   - [Stage 2: VTC (Voice Type Classification)](#stage-2-vtc--voice-type-classification)
   - [Stage 3: SNR (Signal-to-Noise Ratio)](#stage-3-snr--signal-to-noise-ratio)
   - [Stage 4: ESC Classification](#stage-4-esc-classification)
   - [Stage 5: Packaging (Clip Tiling + WebDataset Sharding)](#stage-5-packaging)
3. [Metadata Schema Reference](#3-metadata-schema-reference)
4. [Storage Format & Size Characteristics](#4-storage-format--size-characteristics)
5. [Dataloading Architecture](#5-dataloading-architecture)
6. [End-to-End Workflow: From Raw Audio to Training Batch](#6-end-to-end-workflow)
7. [Extensibility & Adding New Features](#7-extensibility--adding-new-features)
8. [Anticipated Questions & Answers](#8-anticipated-questions--answers)

---

## 1. Pipeline Overview

DL++ separates speech data preparation into two phases:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    OFFLINE — Feature Extraction Pipeline                  │
│                           (runs once per dataset)                         │
│                                                                          │
│   Raw Audio ──► VAD ──► VTC ──► SNR ──► Noise ──► Package               │
│                                                                          │
│   Each stage:                                                            │
│     • Reads audio files listed in a manifest CSV                         │
│     • Runs a GPU/CPU model to extract metadata                           │
│     • Writes structured output (Parquet tables + per-file NPZ arrays)    │
│     • Is fully resumable — skips already-processed files on restart      │
│                                                                          │
│   Package stage:                                                         │
│     • Joins all metadata ("Big Join" on wav_id)                          │
│     • Tiles long-form audio into fixed-duration clips (~600s max)        │
│     • Writes WebDataset TAR shards (FLAC audio + JSON metadata per clip) │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    ONLINE — Dataloading (runs every training step)        │
│                           dataloader/ package                            │
│                                                                          │
│   WebDataset shards ──► FeatureLoader ──► DataProcessor ──► Collator     │
│                                                 │                        │
│                                          ┌──────▼──────┐                 │
│                                          │  DataBatch   │                │
│                                          │  (B,C,T)     │                │
│                                          └─────────────┘                 │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key design principle**: All expensive computation happens offline. The online
dataloading path only does lightweight I/O (reading pre-tiled FLAC clips from TAR
shards) and fast in-memory transforms (resampling, label encoding, mask generation).

### Pipeline Execution Order

All stages must run sequentially (each depends on the previous):

| Order | Stage       | Compute | Typical Rate              |
|-------|-------------|---------|---------------------------|
| 0     | Normalize   | CPU     | Instant (~seconds)        |
| 1     | **VAD**     | CPU     | ~820 MB/s (32 workers)    |
| 2     | **VTC**     | GPU     | ~50 MB/s per GPU          |
| 3     | **SNR**     | GPU     | ~30–50 MB/s per GPU       |
| 4     | **Noise**   | GPU     | ~20–40 MB/s per GPU       |
| 5     | **Package** | CPU     | Single-process + threaded I/O |

SNR and Noise can run in parallel (no dependency between them), but both require
VAD to have completed.

---

## 2. Feature Extraction Stages

### Stage 1: VAD — Voice Activity Detection

| Property | Value |
|----------|-------|
| **Model** | TenVAD (Meta's internal VAD) |
| **Sample Rate** | 16 kHz (auto-resampled) |
| **Hop Size** | 256 samples (~16 ms per frame) |
| **Threshold** | 0.5 (sigmoid probability) |
| **Parallelization** | CPU multiprocessing (`ProcessPoolExecutor`, up to 32 workers) |

**What it produces:**

Per-file metadata (scalar statistics):
- `file_id` — unique identifier (filename stem)
- `duration` — total file length in seconds
- `original_sr` — source sample rate
- `speech_ratio` — fraction of frames above threshold (0–1)
- `n_speech_segments`, `n_silence_segments` — segment counts
- Speech/silence duration statistics: `max`, `min`, `sum`, `avg`, `num` for each
- `has_long_segment` — flag if any speech segment ≥10s exists

Per-file segments (event-level):
- `uid`, `onset`, `offset`, `duration` — speech onset/offset in seconds

**Output files:**
```
output/{dataset}/
├── vad_meta/metadata.parquet          # Per-file scalar stats
├── vad_raw/segments.parquet           # Raw VAD segments (all files concatenated)
├── vad_merged/segments.parquet        # Merged segments (gaps ≤0.1s filled, segs <0.1s removed)
└── vad_probs/{uid}-probs.npz          # [Optional] Per-frame probabilities (float32)
```

**Implementation note:** Audio is processed in streaming blocks (~10 min / 9.6M
samples each) to keep memory constant at ~19 MB per worker regardless of file
length. Checkpoints are saved every 5000 files for crash recovery.

---

### Stage 2: VTC — Voice Type Classification

| Property | Value |
|----------|-------|
| **Model** | Segma (pyannote-derived architecture, custom checkpoint) |
| **Labels** | FEM (female adult), MAL (male adult), KCHI (key child), OCH (other child) |
| **Threshold** | 0.5 per-label sigmoid (multi-label, not mutually exclusive) |
| **Parallelization** | SLURM arrays + GPU batch inference (auto batch-size by VRAM) |

**Batch-size auto-detection:**
- H100 (94 GB) → batch_size=512
- A40 (48 GB) → batch_size=256
- T4 (16 GB) → batch_size=64

**What it produces:**

Per-file metadata (scalar statistics):
- `uid` — unique identifier
- `vtc_threshold` — threshold applied
- `vtc_speech_dur` — total speech duration across all labels
- `vtc_n_segments` — total segment count
- `vtc_label_counts` — JSON dict: `{"FEM": 45, "KCHI": 12, ...}`
- `vtc_max_sigmoid`, `vtc_mean_sigmoid` — model confidence statistics

Per-file segments (event-level):
- `uid`, `onset`, `offset`, `duration`, `label` — each labeled speech segment

**Output files:**
```
output/{dataset}/
├── vtc_meta/shard_{id}.parquet        # Per-file VTC stats (one parquet per SLURM task)
├── vtc_raw/shard_{id}.parquet         # Raw VTC segments with labels
├── vtc_merged/shard_{id}.parquet      # Merged segments (gaps ≤0.3s filled, segs <0.1s removed)
└── logits/{uid}-logits_dict_t.pt      # [Optional] Per-label raw logits (dict → tensor)
```

---

### Stage 3: SNR — Signal-to-Noise Ratio

| Property | Value |
|----------|-------|
| **Model** | Brouhaha (pyannote-based multi-task model) |
| **Outputs** | SNR (dB), C50 clarity (dB), VAD probability — all per frame |
| **Frame Step** | ~16 ms (model-dependent) |
| **Pool Window** | 1.0 s default (averages per-frame values into fixed bins) |
| **Parallelization** | SLURM arrays + GPU batch inference |

**What it produces:**

Per-file metadata (scalar statistics):
- `uid`, `snr_status`, `duration`
- `n_raw_frames`, `n_speech_frames` — frame counts
- `speech_fraction` — fraction above Brouhaha's own VAD threshold
- `snr_mean`, `snr_std`, `snr_min`, `snr_max` — dB values (speech-masked)
- `c50_mean`, `c50_std`, `c50_min`, `c50_max` — clarity in dB

Per-file arrays (frame-level, stored as compressed NPZ):
- `snr` — `(n_frames,)` float16 — per-frame SNR in dB
- `c50` — `(n_frames,)` float16 — per-frame C50 in dB
- `vad` — `(n_frames,)` float16 — VAD probability
- `step_s` — scalar — frame step in seconds
- `vad_threshold` — scalar — 0.5

**Output files:**
```
output/{dataset}/
├── snr_meta/shard_{id}.parquet        # Per-file SNR scalar stats
└── snr/{uid}.npz                      # Per-file frame-level arrays (compressed)
```

---

### Stage 4: ESC Classification

| Property | Value |
|----------|-------|
| **Model** | PANNs CNN14 (AudioSet pre-trained) |
| **Class Count** | 527 AudioSet classes → mapped to 16 semantic categories |
| **Inference Window** | 10 seconds |
| **Native Rate** | 32 kHz |
| **Pool Window** | 1.0 s default |
| **Parallelization** | SLURM arrays + GPU batch inference |

**Semantic categories** (527 AudioSet classes → 16 groups):

| Category | Example Classes |
|----------|----------------|
| laughter | Laughter, baby laughter, giggle |
| crying | Crying, sobbing, whimper |
| singing | Singing, choir, yodeling |
| human_activity | Cough, snoring, footsteps, clapping |
| animal | Dog bark, bird song, cat meow |
| music | All genres + instruments |
| nature | Wind, rain, thunder, ocean |
| vehicle | Cars, trains, aircraft, boats |
| machinery | Chainsaw, drill, engine |
| household | Doors, kitchen sounds, vacuum |
| alarm_signal | Phone ring, siren, beep |
| impact | Gunshot, crash, explosion |
| silence | Silence, sine wave |
| environment | Room reverb, echo, white noise |
| tv_radio | TV, radio, field recording |
| other | Unmapped indices |

**What it produces:**

Per-file metadata (scalar statistics):
- `uid`, `esc_status`, `duration`
- `n_inference_windows`, `n_pooled_bins`
- `dominant_category`, `dominant_prob`
- `prob_{category}` — per-category mean probability (16 columns)

Per-file arrays (window-level, stored as compressed NPZ):
- `categories` — `(n_bins, 16)` float16 — category probabilities per time bin
- `audioset_probs` — `(n_bins, 527)` float16 — raw 527-class probabilities
- `category_names` — `(16,)` string array
- `audioset_names` — `(527,)` string array
- `pool_step_s`, `inference_step_s`, `n_inference_windows` — scalars

**Output files:**
```
output/{dataset}/
├── esc_meta/shard_{id}.parquet      # Per-file ESC scalar stats
└── esc/{uid}.npz                    # Per-file window-level arrays (compressed)
```

---

### Stage 5: Packaging

The packaging stage joins all metadata and tiles long-form audio into clips
suitable for model training.

#### Clip Building Algorithm

1. **Plan**: `n_clips = ceil(duration / max_clip_s)` (default max 600s ≈ 10 min)
2. **Ideal length**: `ideal_step = duration / n_clips`
3. **Adaptive cut-finding** — 6-tier fallback priority:

| Tier | Strategy | Description |
|------|----------|-------------|
| 1 | Long silence gap | ≥10s gap in union of VAD ∪ VTC |
| 2 | Any silence gap | Any gap in VAD ∪ VTC union |
| 3 | VAD-only gap | Gap in VAD where VTC is still active |
| 4 | VTC-only gap | Gap in VTC where VAD is still active |
| 5 | Speaker boundary | Forced cut at VTC speaker transition |
| 6 | Hard cut | No speech gap found — cuts mid-audio |

4. **Grid snapping**: Cuts rounded to nearest segma chunk boundary (3.98s grid)
   for VTC prediction consistency.

#### Per-Clip Metadata (JSON, embedded in each TAR entry)

Each clip in the WebDataset shard includes a JSON sidecar with:

```json
{
  "uid": "source_file_stem",
  "clip_idx": 0,
  "clip_id": "source_file_stem_0000",
  "abs_onset": 0.0,
  "abs_offset": 589.04,
  "duration": 589.04,
  "vtc_speech_duration": 280.4,
  "speech_density": 0.476,
  "n_vtc_segments": 79,
  "n_turns": 53,
  "labels_present": ["FEM", "KCHI", "MAL", "OCH"],
  "has_adult": true,
  "child_speech_duration": 128.9,
  "dominant_label": "FEM",
  "label_durations": {"FEM": 150.4, "MAL": 1.1, "KCHI": 123.8, "OCH": 5.1},
  "vad_speech_duration": 125.9,
  "vad_vtc_iou": 0.423,
  "snr_mean": 11.0,
  "snr_std": 13.2,
  "dominant_esc": "other",
  "vad_segments": [{"onset": 0.0, "offset": 5.3, "duration": 5.3}, ...],
  "vtc_segments": [{"onset": 0.5, "offset": 3.2, "label": "FEM"}, ...]
}
```

All segment timestamps are **relative to clip start** (0 = beginning of clip).

#### Output Files

```
output/{dataset}/shards/
├── shards-000000.tar                  # WebDataset tar (≤100 clips each)
├── shards-000001.tar
├── ...
└── manifest.csv                       # Clip-level metadata (40 columns, 1 row/clip)
```

#### Manifest Columns (40 total)

The clip manifest provides the "Big Join" — all feature outputs unified per clip:

| Group | Columns |
|-------|---------|
| **Identity** | `uid`, `clip_idx`, `clip_id` |
| **Temporal** | `abs_onset`, `abs_offset`, `duration` |
| **VTC** | `vtc_speech_duration`, `vtc_speech_density`, `n_vtc_segments`, `mean_vtc_seg_duration`, `mean_vtc_gap`, `n_turns`, `n_labels`, `labels_present`, `has_adult`, `child_speech_duration`, `adult_speech_duration`, `child_fraction`, `dominant_label` |
| **VAD** | `vad_speech_duration`, `vad_speech_density`, `n_vad_segments`, `vad_vtc_iou` |
| **SNR** | `snr_mean`, `snr_std`, `snr_min`, `snr_max`, `snr_step_s`, `c50_mean`, `c50_std`, `c50_min`, `c50_max` |
| **Noise** | `dominant_esc` |
| **Per-label** | `dur_FEM`, `dur_MAL`, `dur_KCHI`, `dur_OCH`, `vad_cov_FEM`, `vad_cov_MAL`, `vad_cov_KCHI`, `vad_cov_OCH` |

---

## 3. Metadata Schema Reference

### Summary of All Outputs by Stage

| Stage | Per-File Scalars (Parquet) | Per-File Arrays (NPZ/PT) | Per-Clip (JSON in TAR) |
|-------|---------------------------|--------------------------|------------------------|
| **VAD** | 13 columns (speech ratio, segment counts, duration stats) | Optional: `(n_frames,)` raw probabilities | VAD segments with onset/offset |
| **VTC** | 7 columns (speech duration, label counts, sigmoid stats) | Optional: per-label logit tensors | VTC segments with speaker labels |
| **SNR** | 12 columns (SNR/C50 mean/std/min/max, frame counts) | `snr`, `c50`, `vad` arrays (float16) | Scalar SNR stats |
| **ESC** | 20 columns (dominant category, per-category probabilities) | `categories` (n_bins×16), `audioset_probs` (n_bins×527) | Dominant ESC category |
| **Package** | 40 columns in clip manifest | — | Full JSON sidecar per clip |

---

## 4. Storage Format & Size Characteristics

### Format Choices

| Data Type | Format | Why |
|-----------|--------|-----|
| Scalar statistics | **Parquet** (zstd compressed) | Columnar, fast filtering, Polars-native |
| Frame-level arrays | **NPZ** (compressed) | Compact float16, fast numpy I/O |
| Tensor data | **.pt** (PyTorch serialization) | No numpy↔torch conversion overhead |
| Clip metadata | **JSON** | Human-readable, embedded in TAR shards |
| Audio clips | **FLAC** (in TAR) | Lossless, ~60% compression vs WAV |
| Config/provenance | **JSON** | Diffable, human-readable |

### Measured Storage Ratios (Real Data)

Based on the **seedlings_10** dataset (52 long-form recordings, ~80 hours total):

| Component | Size | % of Total |
|-----------|------|------------|
| **Source audio** (WAV, 16-bit PCM) | ~77 GB | — |
| **WebDataset shards** (FLAC in TAR) | **79 GB** | 98.9% |
| **All metadata combined** | **482 MB** | 0.6% |
| — SNR arrays (NPZ) | 155 MB | 0.19% |
| — Noise arrays (NPZ) | 266 MB | 0.33% |
| — VAD segments (Parquet) | 24 MB | 0.03% |
| — VTC segments (Parquet) | 11 MB | 0.01% |
| — Scalar metadata (all Parquet) | 80 KB | <0.01% |
| — Clip manifest (CSV) | 1.7 MB | <0.01% |
| **Total output** | ~80 GB | — |

**Key takeaway**: Metadata overhead is **<1%** of waveform data. The dominant
storage cost is always the audio itself. Even the heaviest metadata (ESC
embeddings with 527 AudioSet probabilities per time bin) adds only ~0.3%.

### Per-File Array Sizes (Typical)

For a ~16-hour long-form recording at 16 kHz:

| Array | Dimensions | dtype | Compressed Size |
|-------|-----------|-------|-----------------|
| SNR `{uid}.npz` | ~3.6M frames × 3 arrays | float16 | ~12–15 MB |
| Noise `{uid}.npz` | ~5.7K bins × 527+16 arrays | float16 | ~4–6 MB |
| VTC logits (optional) | ~3.6M frames × 4 labels | float32 | ~25–30 MB |
| VAD probs (optional) | ~3.6M frames × 1 | float32 | ~10–14 MB |

---

## 5. Dataloading Architecture

### Design Goals

1. **Zero-config for common case**: Load shards, get batched tensors — no manual metadata wrangling
2. **Composable transforms**: Plug in resampling, segmentation, label encoding, masking as needed
3. **Distributed-native**: Split shards across nodes/workers automatically via WebDataset
4. **Metadata always available**: Per-sample metadata travels with the waveform through the entire pipeline

### Component Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  WebDatasetSpeechDataset (IterableDataset)                      │
│                                                                 │
│  shard_urls = ["shards/shards-{000000..000050}.tar"]            │
│                                                                 │
│  ┌──────────────┐   For each sample in shard:                   │
│  │  Decode FLAC  │──► waveform (channels, samples)              │
│  │  + Parse JSON │──► metadata dict (40+ fields)                │
│  └──────┬───────┘                                               │
│         │                                                       │
│  ┌──────▼───────┐   Composable pipeline:                        │
│  │  DataProcessor│   Resampler(target_sr=16000)                  │
│  │  (Compose)    │──►VADSegmenter(segments_key="vad_segments")  │
│  │               │──►LabelEncoder(labels=["FEM","MAL",...])     │
│  │               │──►MaskGenerator(frame_shift_s=0.02)          │
│  └──────┬───────┘                                               │
│         │                                                       │
│  ┌──────▼───────┐   Batching:                                   │
│  │ SpeechCollator│   Pad to max length, build attention masks    │
│  │               │   Extract tensor fields (SNR, frame labels)  │
│  └──────┬───────┘                                               │
│         │                                                       │
│  ┌──────▼───────┐                                               │
│  │   DataBatch   │   waveforms: (B, 1, T_max) float32          │
│  │               │   attention_mask: (B, T_max) bool            │
│  │               │   snr_db: (B,) float32                       │
│  │               │   frame_labels: (B, T_frames, 4) bool        │
│  │               │   wav_ids: list[str]                          │
│  │               │   metadata: list[dict]                        │
│  └───────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Distributed Training Support

The `WebDatasetSpeechDataset` uses WebDataset's native sharding:

```python
wds.WebDataset(urls, resampled=True)          # Infinite epoch streaming
    .compose(wds.shardlists.split_by_node)    # Multi-node: each node sees different shards
    .compose(wds.shardlists.split_by_worker)  # Multi-worker: each DataLoader worker sees different shards
    .shuffle(buffer_size)                      # In-memory sample shuffle
```

This means **no manual shard assignment** is needed — DDP and DataLoader workers
automatically partition the data.

An `EvalSpeechDataset` (map-style `Dataset`) is also provided for deterministic
evaluation with fixed iteration order.

### Minimal Usage Example

```python
from dataloader import (
    WebDatasetSpeechDataset,
    SpeechCollator,
    Compose,
    Resampler,
    LabelEncoder,
    MaskGenerator,
)

# 1. Define transforms
processor = Compose([
    Resampler(target_sr=16_000),
    LabelEncoder(labels=["FEM", "MAL", "KCHI", "OCH"], segments_key="vtc_segments"),
    MaskGenerator(segments_key="vtc_segments", frame_shift_s=0.02, num_labels=4),
])

# 2. Create dataset (reads shards, applies transforms)
dataset = WebDatasetSpeechDataset(
    shard_urls="output/seedlings_10/shards/shards-{000000..000050}.tar",
    audio_key="flac",
    metadata_keys=["json"],
    processor=processor,
    shuffle_buffer=1000,
)

# 3. Create DataLoader with collator
collator = SpeechCollator(pad_to_multiple_of=512)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collator,
    num_workers=4,
)

# 4. Training loop — one line to get a batch
for batch in loader:
    batch.to("cuda")
    model(batch.waveforms, batch.attention_mask, batch.frame_labels)
```

### Available Runtime Transforms

| Transform | Purpose | Input → Output |
|-----------|---------|----------------|
| `Resampler` | Change sample rate | waveform at any SR → waveform at target SR |
| `VADSegmenter` | Extract speech-only audio | full waveform → concatenated speech segments |
| `Normalizer` | Volume normalization | waveform → RMS or peak normalized waveform |
| `LabelEncoder` | String labels → integers | metadata with label strings → label IDs + vocab |
| `MaskGenerator` | Frame-level multi-hot masks | metadata with segments → attention + label tensors |
| `Denoiser` | Spectral gating noise removal | noisy waveform → denoised waveform |
| `WaveformProcessor` | Custom waveform transforms (dual-mode: online or cached offline) | waveform → transformed waveform |

---

## 6. End-to-End Workflow

### For the Pipeline Operator (Feature Extraction)

```bash
# 1. Standardize the manifest
uv run python -m src.pipeline.normalize --dataset my_corpus --path-col audio_path

# 2. Run feature extraction via SLURM
sbatch slurm/vad.slurm my_corpus                   # CPU multiprocessing
sbatch --array=0-3 slurm/vtc.slurm my_corpus       # GPU, 4 SLURM tasks
sbatch --array=0-1 slurm/snr.slurm my_corpus       # GPU, 2 tasks
sbatch --array=0-1 slurm/esc.slurm my_corpus     # GPU, 2 tasks (parallel with SNR)

# 3. Package into WebDataset shards
sbatch slurm/package.slurm my_corpus               # CPU, single process

# Result: output/my_corpus/shards/*.tar + manifest.csv
```

### For the ML Engineer (Training)

```python
# Just point at the shards — everything else is handled
dataset = WebDatasetSpeechDataset(
    shard_urls="output/my_corpus/shards/shards-{000000..NNNNNN}.tar",
    audio_key="flac",
    metadata_keys=["json"],
    processor=my_transforms,
)
```

The shard manifest (`manifest.csv`) can also be used for:
- **Stratified sampling**: Filter clips by `speech_density`, `snr_mean`, `dominant_esc`, etc.
- **Quality gating**: Exclude clips below an SNR threshold or with too little speech
- **Curriculum learning**: Sort by difficulty metrics

---

## 7. Extensibility & Adding New Features

### Adding a New Feature Processor

The `FeatureProcessor` ABC provides a standard interface:

```python
class FeatureProcessor(ABC):
    name: ClassVar[str]
    version: ClassVar[str]

    def process(self, wav_id, audio_path) -> MetadataDict: ...
    def save(self, wav_id, metadata, output_dir) -> Path: ...
    def load(self, wav_id, output_dir) -> MetadataDict: ...
    def exists(self, wav_id, output_dir) -> bool: ...
```

New processors (e.g., phoneme alignment, emotion detection, speaker embeddings)
implement this interface and register via `@ProcessorRegistry.register`. The
packaging stage picks up new metadata automatically via the manifest joiner.

### Adding a New Transform

```python
class MyTransform(DataProcessor):
    def __call__(self, waveform, sample_rate, metadata):
        # Your transform logic
        return waveform, sample_rate, metadata
```

Drop it into a `Compose` pipeline alongside existing transforms.

### Dual-Mode Waveform Processing

The `WaveformProcessor` base class supports:
- **Online mode**: Transform applied at every dataload (no disk writes)
- **Offline mode**: Transform applied once, result cached to disk with provenance
  metadata (processor name, version, parameters, timestamp). Subsequent loads skip
  the transform and read the cached file.

---

## 8. Anticipated Questions & Answers

### Architecture & Design

**Q: Why not process everything in a single pass?**  
A: Each model (TenVAD, segma, Brouhaha, PANNs) has different compute
requirements (CPU vs GPU, different sample rates, different batch sizes). Separate
stages allow independent scaling and resumability. A VTC failure doesn't require
re-running VAD.

**Q: Why WebDataset TAR shards instead of HDF5 / TFRecord / individual files?**  
A: WebDataset TARs provide sequential I/O (essential for spinning disks and distributed
filesystems), native PyTorch `IterableDataset` support, built-in multi-node/multi-worker
sharding, and trivial shuffling via shard permutation + buffer.

**Q: How do you handle multi-node distributed training?**  
A: WebDataset's `split_by_node` and `split_by_worker` automatically partition
shards so each rank sees unique data. With `resampled=True`, epoch boundaries
are handled seamlessly for infinite streaming.

**Q: What's the "Big Join" and why is it needed?**  
A: Each feature processor outputs metadata in its own format (different parquet
files, different column names). The Big Join unifies everything by `wav_id` into
a single manifest. This happens at packaging time — by training time, all metadata
is already pre-joined in the JSON sidecar of each clip.

**Q: How does this integrate with Meta's existing training infrastructure (metasr-internal / fs2)?**  
A: We've designed a compatibility shim layer (`dataloader/compat/`, in progress).
Our `DataBatch`, `SpeechCollator`, and `SpeechDataset` have clean, documented
interfaces. When upstream type signatures are available, we adapt the shim without
rewriting core logic.

### Performance & Scalability

**Q: What's the metadata overhead relative to waveform storage?**  
A: <1%. On our 80-hour test dataset (52 files), total metadata is 482 MB vs 79 GB
of FLAC audio. Even the most verbose metadata (ESC: 527-class probabilities at
1-second resolution) adds only 0.3%.

**Q: How large are the WebDataset shards?**  
A: Typically 200–900 MB each (configurable via `max_clips_per_shard`, default 100).
Each shard contains FLAC audio + JSON metadata for ~100 clips of ≤10 minutes each.

**Q: What's the bottleneck in the extraction pipeline?**  
A: For most datasets, VTC (GPU-bound). VAD is fast on CPU (820 MB/s). SNR and Noise
are lighter GPU tasks. The preflight tool (`src/pipeline/preflight.py`) estimates
wall-clock time and recommends SLURM array sizes based on dataset size and available
GPUs.

**Q: Can I add more GPU nodes to speed up extraction?**  
A: Yes. VTC, SNR, and Noise all support SLURM array parallelism — just increase
`--array_count`. Each task processes a disjoint shard of the manifest.

**Q: What happens if a job fails mid-way?**  
A: All stages are fully resumable. They detect previously-processed files and skip
them on restart. VAD also saves periodic checkpoints (every 5000 files).

### Data Quality & Reproducibility

**Q: Is the pipeline deterministic?**  
A: Yes. All random seeds are set to 42 (`random`, `numpy`, `torch`, plus
`cudnn.deterministic=True`, `cudnn.benchmark=False`). Tests verify bit-identical
results across reruns for both VAD and VTC.

**Q: How are clip boundaries chosen? Do you cut mid-sentence?**  
A: The adaptive 6-tier algorithm tries to cut at silence gaps first (longest gaps
preferred). Only as a last resort (Tier 6) does it cut mid-audio. Cuts are also
snapped to the segma grid (3.98s) so VTC predictions align with chunk boundaries.

**Q: What's the VAD/VTC IoU and why does it matter?**  
A: IoU (Intersection over Union) between VAD-detected speech and VTC-detected
speech measures agreement between the two models. High IoU (>0.8) means the models
agree on where speech occurs. Low IoU can indicate model disagreement on overlapping
speech, background noise, or non-speech vocalizations.

**Q: How do you handle overlapping speakers (e.g., two people talking at once)?**  
A: VTC uses multi-label classification — each frame can have multiple active
labels (FEM + KCHI simultaneously). VAD captures total speech energy including
overlaps. The `vad_vtc_iou` metric reflects how well these align.

### Metadata & Features

**Q: What speaker labels are available?**  
A: Four classes: FEM (female adult), MAL (male adult), KCHI (key child), OCH
(other child). These are defined in the segma model config and can be extended
by retraining the VTC model.

**Q: Can I filter clips by acoustic quality before training?**  
A: Yes. The clip manifest includes `snr_mean`, `speech_density`, `n_turns`,
`dominant_esc`, and per-label durations — all usable as filter predicates.

**Q: Why store 527 AudioSet probabilities instead of just the dominant category?**  
A: The full probability vector enables: (1) custom category groupings post-hoc,
(2) multi-label ESC (music + speech + traffic simultaneously),
(3) fine-grained quality filtering (e.g., exclude clips where `music > 0.3`).

**Q: What is the C50 metric from the SNR stage?**  
A: C50 (Clarity) is the early-to-late energy ratio in dB — it measures room
reverberation. Higher C50 means clearer, less reverberant audio. Useful for
filtering out recordings with extreme echo.

### Usage & Integration

**Q: What's the minimum I need to start training?**  
A: Just the TAR shards. Create a `WebDatasetSpeechDataset` with the shard paths
and a `SpeechCollator`. You get `DataBatch` objects with padded waveforms,
attention masks, and all metadata. Transforms are optional.

**Q: Can I use raw (unpackaged) metadata without creating shards?**  
A: Yes. The `MetadataLoader` and `WaveformLoader` classes can read directly from
the pipeline outputs (parquet + npz files). The `MetadataStore` backends handle
Parquet, NPZ, JSON, and PyTorch formats. This is useful for exploration and
prototyping before committing to the packaging step.

**Q: Can I add custom metadata to the pipeline?**  
A: Implement a `FeatureProcessor` subclass with your extraction logic, register
it, and it participates in the Big Join automatically. At training time, the
metadata flows through to `DataBatch.metadata` as a dict.

**Q: What about phoneme alignments, emotion, or speaker embeddings?**  
A: The architecture supports these as additional `FeatureProcessor` implementations.
Phoneme alignments are explicitly planned but deferred to a future phase. The
infrastructure (processor interface, registry, manifest joining, metadata storage)
is already in place.

---

## Appendix: Quick Reference Card

### Models Used

| Stage | Model | License | GPU Required |
|-------|-------|---------|-------------|
| VAD | TenVAD | Meta (internal) | No (CPU) |
| VTC | Segma (custom checkpoint) | Research | Yes |
| SNR | Brouhaha | MIT | Yes |
| Noise | PANNs CNN14 | MIT | Yes |

### Key Identifiers

| ID | Format | Example | Scope |
|----|--------|---------|-------|
| `wav_id` / `uid` | Filename stem | `Bergelson_9_9457` | Entire pipeline |
| `clip_id` | `{wav_id}_{clip_idx:04d}` | `Bergelson_9_9457_0000` | Post-packaging |

### Key File Paths

```
manifests/{dataset}.csv                # Input: normalized manifest with absolute paths
output/{dataset}/vad_*/                # VAD outputs
output/{dataset}/vtc_*/                # VTC outputs
output/{dataset}/snr*/                 # SNR outputs
output/{dataset}/esc*/               # ESC outputs
output/{dataset}/shards/               # Final WebDataset shards
output/{dataset}/shards/manifest.csv   # Clip-level metadata (Big Join result)
```
