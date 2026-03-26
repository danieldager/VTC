"""Dataloader++ — Feature loader and data processing framework.

This package implements the Dataloader++ specification for Meta's speech
training infrastructure. It provides:

- **Feature Processor** abstractions for offline metadata extraction
- **Feature Loader** abstractions for waveform + metadata I/O
- **Manifest management** with typed schemas and multi-source joins
- **Runtime transforms** (segment, resample, encode, mask)
- **Collation** into typed :class:`DataBatch` containers

See ``docs/DATALOADER_DESIGN.md`` for the full design document.

Quick Start
-----------
::

    from dataloader import (
        Compose, DataBatch, DataProcessor, FeatureLoader,
        FeatureProcessor, ManifestJoiner, MetadataManifest,
        SpeechCollator, SpeechDataset,
    )
"""

from dataloader.batch.base import Collator
from dataloader.batch.data_batch import DataBatch
from dataloader.batch.speech import SpeechCollator
from dataloader.loader.base import FeatureLoader
from dataloader.loader.metadata import MetadataLoader
from dataloader.loader.waveform import WaveformLoader
from dataloader.manifest.joiner import ManifestJoiner
from dataloader.manifest.schema import MetadataManifest
from dataloader.manifest.store import (
    JsonStore,
    MetadataStore,
    NpzStore,
    ParquetStore,
)
from dataloader.processor.base import FeatureProcessor
from dataloader.processor.registry import ProcessorRegistry
from dataloader.transform.audio import Normalizer, Resampler, VADSegmenter
from dataloader.transform.base import Compose, DataProcessor
from dataloader.transform.label import LabelEncoder, MaskGenerator

__all__ = [
    # Processor
    "FeatureProcessor",
    "ProcessorRegistry",
    # Loader
    "FeatureLoader",
    "MetadataLoader",
    "WaveformLoader",
    # Manifest
    "ManifestJoiner",
    "MetadataManifest",
    "MetadataStore",
    "ParquetStore",
    "NpzStore",
    "JsonStore",
    # Transform
    "Compose",
    "DataProcessor",
    "LabelEncoder",
    "MaskGenerator",
    "Normalizer",
    "Resampler",
    "VADSegmenter",
    # Batch
    "Collator",
    "DataBatch",
    "SpeechCollator",
]
