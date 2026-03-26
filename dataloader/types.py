"""Shared type aliases and enums for the dataloader package.

These types form the lingua franca between processors, loaders, transforms,
and collators. All public APIs in this package use these aliases rather than
raw primitives so that type changes propagate automatically.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import numpy as np
import torch

# ── Identifiers ──────────────────────────────────────────────────────────────
WavID: TypeAlias = str
"""Unique waveform identifier — typically the stem of the audio filename."""

ClipID: TypeAlias = str
"""Clip identifier in the format ``{wav_id}_{clip_idx:04d}``."""

# ── Audio ─────────────────────────────────────────────────────────────────────
Waveform: TypeAlias = torch.Tensor
"""Audio tensor of shape ``(channels, samples)``, dtype ``float32``."""

SampleRate: TypeAlias = int
"""Integer sample rate in Hz (e.g. ``16_000``)."""

# ── Metadata ──────────────────────────────────────────────────────────────────
MetadataDict: TypeAlias = dict[str, object]
"""Arbitrary key → value metadata for a single sample.

Values may be scalars, lists, numpy arrays, or nested dicts. The schema is
defined by each :class:`FeatureProcessor` and validated during the Big Join.
"""

SegmentList: TypeAlias = list[dict[str, float]]
"""List of segments, each a dict with at least ``onset`` and ``offset`` keys.

Optional keys include ``duration`` (derived) and ``label`` (speaker tag).
Example::

    [
        {"onset": 1.2, "offset": 5.4, "label": "FEM"},
        {"onset": 6.0, "offset": 9.1, "label": "KCHI"},
    ]
"""

# ── Tensors ───────────────────────────────────────────────────────────────────
Mask: TypeAlias = torch.BoolTensor
"""Boolean mask tensor, typically shape ``(batch, time)``."""

LabelTensor: TypeAlias = torch.LongTensor
"""Integer label tensor, typically shape ``(batch, max_segments)``."""


# ── Enums ─────────────────────────────────────────────────────────────────────
class SpeakerLabel(str, Enum):
    """Canonical speaker labels used by the VTC pipeline."""

    FEM = "FEM"
    MAL = "MAL"
    KCHI = "KCHI"
    OCH = "OCH"


class JoinStrategy(str, Enum):
    """Join strategy for :class:`ManifestJoiner`."""

    INNER = "inner"
    LEFT = "left"
    OUTER = "outer"


class MetadataFormat(str, Enum):
    """Supported storage formats for :class:`MetadataStore` backends."""

    PARQUET = "parquet"
    NPZ = "npz"
    JSON = "json"
    PT = "pt"
