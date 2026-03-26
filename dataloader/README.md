# Dataloader++

Feature loader and data processing framework for Meta's speech training
infrastructure. See [`docs/DATALOADER_DESIGN.md`](../docs/DATALOADER_DESIGN.md)
for the full design document.

## Quick Start

```python
from dataloader import (
    Collator,
    Compose,
    DataBatch,
    DataProcessor,
    FeatureLoader,
    FeatureProcessor,
    ManifestJoiner,
    MetadataManifest,
    MetadataStore,
    SpeechDataset,
)
```

## Package Structure

| Module | Purpose |
|---|---|
| `processor/` | Feature Processor ABCs — offline metadata extraction |
| `loader/` | Feature Loader ABCs — waveform + metadata I/O |
| `manifest/` | Manifest schema, Big Join, unified metadata store |
| `transform/` | Runtime data transforms (segment, resample, encode) |
| `batch/` | Collation and `DataBatch` containers |
| `dataset/` | PyTorch Dataset implementations |
| `types.py` | Shared type aliases and enums |
