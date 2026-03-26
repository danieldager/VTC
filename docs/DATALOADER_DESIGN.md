# Dataloader++ Design Document

> **Status**: Phase 1 — Interface Design  
> **Authors**: DL++ Team  
> **Last Updated**: 2026-03-26

---

## 1. Motivation

The DL++ pipeline (`src/pipeline/`) already functions as a **Feature Processor**: it
ingests raw long-form audio, runs GPU-accelerated models (TenVAD, segma, Brouhaha,
PANNs), and produces rich per-file metadata in standardized formats. The packaging
step (`src/pipeline/package.py`) tiles files into clips and writes WebDataset shards.

The **Dataloader++** initiative at Meta requires a Feature Loader counterpart
that can:

1. Load waveforms and pre-computed metadata from these shards (or raw outputs).
2. Join heterogeneous metadata manifests by waveform ID.
3. Apply runtime transforms (segmentation, resampling, label encoding, masking).
4. Collate variable-length samples into batched tensors for model training.

This document specifies the design for a new `dataloader/` package that implements
these capabilities without modifying the existing `src/` pipeline.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (Feature Processor)                   │
│                                                                      │
│  Raw Audio ──► VAD ──► VTC ──► SNR ──► Noise ──► Package            │
│               (src/pipeline/*)                                       │
│                                                                      │
│  Outputs:   vad_{raw,merged}/segments.parquet                        │
│             vtc_{raw,merged}/shard_*.parquet                         │
│             snr/{uid}.npz   noise/{uid}.npz                          │
│             shards/*.tar    stats/*.parquet                          │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        ONLINE  (Feature Loader)                      │
│                        dataloader/                                   │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐             │
│  │  Manifest    │──►│  Feature      │──►│  Data         │             │
│  │  Joiner      │   │  Loader       │   │  Processor    │             │
│  └─────────────┘   └──────────────┘   └──────┬────────┘             │
│                                               │                      │
│                                        ┌──────▼────────┐             │
│                                        │  Collator      │             │
│                                        └──────┬────────┘             │
│                                               │                      │
│                                        ┌──────▼────────┐             │
│                                        │  DataBatch     │             │
│                                        └───────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Package Layout

```
dataloader/
├── __init__.py                 # Public API surface
├── py.typed                    # PEP 561 type-checking marker
├── README.md                   # Package documentation
│
├── types.py                    # Shared type aliases and enums
│
├── processor/                  # Feature Processor abstractions
│   ├── __init__.py
│   ├── base.py                 # FeatureProcessor ABC
│   └── registry.py             # Processor discovery & registration
│
├── loader/                     # Feature Loader abstractions
│   ├── __init__.py
│   ├── base.py                 # FeatureLoader ABC
│   ├── waveform.py             # WaveformLoader (audio I/O)
│   └── metadata.py             # MetadataLoader (JSON/Parquet/NPZ)
│
├── manifest/                   # Manifest management
│   ├── __init__.py
│   ├── schema.py               # MetadataManifest schema definition
│   ├── joiner.py               # ManifestJoiner (Big Join)
│   └── store.py                # MetadataStore (unified I/O)
│
├── transform/                  # Runtime data transforms
│   ├── __init__.py
│   ├── base.py                 # DataProcessor ABC + Compose
│   ├── audio.py                # Audio transforms (resample, segment, normalize)
│   └── label.py                # Label transforms (encode, mask generation)
│
├── batch/                      # Batching and collation
│   ├── __init__.py
│   ├── base.py                 # Collator ABC
│   ├── data_batch.py           # DataBatch container
│   └── speech.py               # SpeechCollator implementation
│
└── dataset/                    # PyTorch Dataset implementations
    ├── __init__.py
    ├── base.py                 # SpeechDataset ABC
    └── webdataset.py           # WebDataset-backed loader
```

---

## 4. Phase 1 — Interface Design

### 4.1 Shared Types (`types.py`)

```python
from enum import Enum
from typing import TypeAlias

import numpy as np
import torch

# ── Identifiers ──────────────────────────────────────────────────────
WavID: TypeAlias = str                             # Unique waveform identifier
ClipID: TypeAlias = str                            # "{wav_id}_{clip_idx:04d}"

# ── Audio ─────────────────────────────────────────────────────────────
Waveform: TypeAlias = torch.Tensor                 # shape (channels, samples)
SampleRate: TypeAlias = int                        # e.g. 16_000

# ── Metadata ──────────────────────────────────────────────────────────
MetadataDict: TypeAlias = dict[str, object]        # Arbitrary key→value metadata
SegmentList: TypeAlias = list[dict[str, float]]    # [{"onset": ..., "offset": ...}, ...]

# ── Tensors ───────────────────────────────────────────────────────────
Mask: TypeAlias = torch.BoolTensor                 # shape (batch, time)
LabelTensor: TypeAlias = torch.LongTensor          # shape (batch, n_segments)
```

### 4.2 FeatureProcessor ABC (`processor/base.py`)

Wraps an offline model/tool that extracts metadata from audio.

```python
class FeatureProcessor(ABC):
    """Base class for offline feature extraction.

    Implementations wrap a model or tool that reads raw audio and produces
    metadata. The processor is responsible for its own parallelization
    strategy (GPU batching, CPU multiprocessing, SLURM arrays).
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def process(self, wav_id: WavID, audio_path: Path) -> MetadataDict: ...

    @abstractmethod
    def save(self, wav_id: WavID, metadata: MetadataDict, output_dir: Path) -> Path: ...

    @abstractmethod
    def load(self, wav_id: WavID, output_dir: Path) -> MetadataDict: ...

    @abstractmethod
    def exists(self, wav_id: WavID, output_dir: Path) -> bool: ...
```

### 4.3 MetadataManifest (`manifest/schema.py`)

A typed wrapper around a Polars DataFrame that enforces a common schema.

```python
class MetadataManifest:
    """Typed manifest of metadata entries keyed by wav_id.

    Wraps a Polars DataFrame with at minimum a `wav_id` column plus
    processor-specific metadata columns. Supports efficient filtering,
    joining, and serialization to Parquet.
    """

    def __init__(self, df: pl.DataFrame, source: str): ...

    @classmethod
    def from_parquet(cls, path: Path, source: str) -> MetadataManifest: ...

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, source: str) -> MetadataManifest: ...

    def to_parquet(self, path: Path) -> None: ...
    def filter(self, predicate: pl.Expr) -> MetadataManifest: ...
    def select(self, columns: list[str]) -> MetadataManifest: ...
    def join(self, other: MetadataManifest, on: str = "wav_id") -> MetadataManifest: ...

    @property
    def wav_ids(self) -> list[WavID]: ...

    @property
    def df(self) -> pl.DataFrame: ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
```

### 4.4 ManifestJoiner (`manifest/joiner.py`)

Declarative multi-manifest join.

```python
class ManifestJoiner:
    """Joins multiple MetadataManifest objects by wav_id.

    Performs an outer join by default, with configurable join strategy
    and column conflict resolution.
    """

    def __init__(self, how: Literal["inner", "left", "outer"] = "inner"): ...

    def add(self, manifest: MetadataManifest) -> ManifestJoiner: ...
    def join(self) -> MetadataManifest: ...
```

### 4.5 MetadataStore (`manifest/store.py`)

Unified read/write interface across storage formats.

```python
class MetadataStore(ABC):
    """Abstract interface for reading and writing per-file metadata.

    Concrete implementations handle Parquet, NPZ, JSON, and .pt formats
    behind a uniform API.
    """

    @abstractmethod
    def load(self, wav_id: WavID) -> MetadataDict: ...

    @abstractmethod
    def save(self, wav_id: WavID, data: MetadataDict) -> None: ...

    @abstractmethod
    def exists(self, wav_id: WavID) -> bool: ...

    @abstractmethod
    def list_ids(self) -> list[WavID]: ...
```

### 4.6 FeatureLoader ABC (`loader/base.py`)

Loads a single sample (waveform + metadata) given a wav_id.

```python
class FeatureLoader(ABC):
    """Base class for loading waveform and metadata for a single sample.

    Subclasses implement format-specific I/O (raw files, WebDataset
    shards, HDF5, etc.).
    """

    @abstractmethod
    def load_waveform(self, wav_id: WavID) -> tuple[Waveform, SampleRate]: ...

    @abstractmethod
    def load_metadata(self, wav_id: WavID) -> MetadataDict: ...

    def load(self, wav_id: WavID) -> tuple[Waveform, SampleRate, MetadataDict]:
        waveform, sr = self.load_waveform(wav_id)
        metadata = self.load_metadata(wav_id)
        return waveform, sr, metadata

    @abstractmethod
    def available_ids(self) -> list[WavID]: ...
```

### 4.7 DataProcessor (`transform/base.py`)

Composable runtime transforms.

```python
class DataProcessor(ABC):
    """Base class for runtime transforms on waveform + metadata."""

    @abstractmethod
    def __call__(
        self, waveform: Waveform, sample_rate: SampleRate, metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]: ...


class Compose(DataProcessor):
    """Chain multiple DataProcessor transforms sequentially."""

    def __init__(self, processors: list[DataProcessor]): ...

    def __call__(
        self, waveform: Waveform, sample_rate: SampleRate, metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]: ...
```

### 4.8 DataBatch (`batch/data_batch.py`)

Typed container for a collated batch.

```python
@dataclass
class DataBatch:
    """Container for a collated batch of speech samples.

    All tensors in a batch share the first dimension (batch_size).
    Variable-length sequences are padded and accompanied by masks.
    """

    waveforms: torch.Tensor           # (B, C, T) padded waveforms
    waveform_lengths: torch.LongTensor # (B,) original lengths in samples
    sample_rate: int
    metadata: list[MetadataDict]       # per-sample metadata (unpadded)
    attention_mask: torch.BoolTensor   # (B, T) True = valid sample

    # Optional fields populated by specific collators
    labels: torch.LongTensor | None = None        # (B, max_segments)
    label_mask: torch.BoolTensor | None = None     # (B, max_segments)
    segments: list[SegmentList] | None = None       # raw segment lists
```

### 4.9 Collator ABC (`batch/base.py`)

```python
class Collator(ABC):
    """Base class for batching and collation."""

    @abstractmethod
    def __call__(
        self, samples: list[tuple[Waveform, SampleRate, MetadataDict]],
    ) -> DataBatch: ...
```

### 4.10 SpeechDataset (`dataset/base.py`)

Bridges the loader + processor into a `torch.utils.data.Dataset`.

```python
class SpeechDataset(Dataset, ABC):
    """Base class for speech datasets.

    Combines a FeatureLoader (I/O) with an optional DataProcessor
    (runtime transforms) into a standard PyTorch Dataset.
    """

    def __init__(
        self,
        loader: FeatureLoader,
        processor: DataProcessor | None = None,
    ): ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Waveform, SampleRate, MetadataDict]: ...
```

---

## 5. Phase 2 — Refactor Existing Stages

Wrap each pipeline stage (`vad.py`, `vtc.py`, `snr.py`, `noise.py`) as a
`FeatureProcessor` subclass. These adapters live in a new `dataloader/adapters/`
directory and import from `src/` without modifying the original code.

```
dataloader/adapters/
├── __init__.py
├── vad.py             # VADProcessor(FeatureProcessor)
├── vtc.py             # VTCProcessor(FeatureProcessor)
├── snr.py             # SNRProcessor(FeatureProcessor)
└── noise.py           # NoiseProcessor(FeatureProcessor)
```

Extract the Big Join logic from `src/pipeline/package.py` into `ManifestJoiner`.

---

## 6. Phase 3 — Build the Feature Loader

Implement concrete loader, processor, and collator subclasses:

| Class | Location | Description |
|-------|----------|-------------|
| `WebDatasetLoader` | `loader/webdataset.py` | Reads `.tar` shards, returns `(waveform, metadata)` |
| `RawFileLoader` | `loader/raw.py` | Reads original audio + side-car metadata |
| `VADSegmenter` | `transform/audio.py` | Segments waveform by VAD timestamps |
| `Resampler` | `transform/audio.py` | Resamples to target sample rate |
| `LabelEncoder` | `transform/label.py` | Speaker labels → integer IDs |
| `MaskGenerator` | `transform/label.py` | Generates attention & prediction masks |
| `SpeechCollator` | `batch/speech.py` | Pads + collates into `DataBatch` |
| `MetadataSampler` | `dataset/sampler.py` | Stratified sampling from stats DataFrames |

---

## 7. Phase 4 — Validation

1. End-to-end test: existing pipeline → shards → Feature Loader → batched tensors
2. Round-trip test: `FeatureProcessor.process()` → `.save()` → `.load()` identity
3. Collation test: variable-length clips → correct padding + masks
4. Benchmark: clips/sec throughput, GPU saturation during training

---

## 8. Design Decisions

### 8.1 Separate Package (`dataloader/`) vs. Extending `src/`

We use a new top-level `dataloader/` directory to:
- Avoid breaking any existing pipeline functionality.
- Allow independent versioning and testing.
- Clarify the boundary: `src/` = offline processing, `dataloader/` = online loading.

### 8.2 Polars for Manifests

All manifest operations use **Polars** (not pandas) for consistency with the existing
codebase and for its superior performance on large-scale joins and filtering.

### 8.3 WebDataset as Primary Shard Format

The existing pipeline already writes WebDataset `.tar` shards. We keep this as the
primary format for streaming I/O compatibility with distributed training and S3.

### 8.4 Adapter Pattern (Not Rewrite)

Phase 2 wraps existing pipeline stages as `FeatureProcessor` subclasses via thin
adapters. The original `src/pipeline/*.py` scripts remain untouched — they continue
to work as standalone CLI tools.

### 8.5 Where This Lives

This is initially built inside the VTC repo as `dataloader/`. It is designed to be
extracted into a standalone package (e.g., `speech-dataloader`) once the API
stabilizes, for import into `metasr-internal` or `fs2`.

---

## 9. Mapping to Dataloader++ Spec

| Spec Concept | VTC Implementation | Status |
|---|---|---|
| Feature Processor (offline) | `src/pipeline/{vad,vtc,snr,noise}.py` | ✅ Exists |
| Parallelized across GPUs/CPUs | SLURM arrays + ProcessPoolExecutor | ✅ Exists |
| Metadata manifests by waveform ID | `output/{dataset}/*/segments.parquet` | ✅ Exists |
| Big Join across manifests | `ManifestJoiner` | 🔨 Phase 1 |
| Waveform Loader | `FeatureLoader` → `WebDatasetLoader` | 🔨 Phase 1–3 |
| Metadata Loader | `MetadataStore` / `MetadataLoader` | 🔨 Phase 1–3 |
| Data Processor | `DataProcessor` → `Compose` | 🔨 Phase 1–3 |
| Batching & Collation | `Collator` → `SpeechCollator` → `DataBatch` | 🔨 Phase 1–3 |
| Extensibility | `FeatureProcessor` ABC + registry | 🔨 Phase 1–2 |

---

## 10. Open Questions

- **Phoneme alignments**: Not currently in the VTC pipeline. A new
  `PhonemeProcessor(FeatureProcessor)` would need a forced aligner (MFA or
  CTC-segmentation). Deferred to Phase 2+.
- **Online vs. offline segmentation**: Current clip tiling is offline. Support for
  model-defined windowing (online) is a Phase 3 concern.
- **Target integration**: `metasr-internal` or `fs2`? The adapter pattern and
  standalone package design supports either.
- **Storage backend**: WebDataset tars vs. HDF5 vs. Lance. Current implementation
  uses tars; the `MetadataStore` abstraction allows adding backends later.
