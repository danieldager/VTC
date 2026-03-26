"""Typed container for a collated batch of speech samples.

:class:`DataBatch` is the output of a :class:`~dataloader.batch.base.Collator`.
It provides a single, typed object that models consume directly from
``torch.utils.data.DataLoader``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dataloader.types import MetadataDict, SegmentList


@dataclass
class DataBatch:
    """Container for a collated batch of speech samples.

    All tensor fields share the first dimension (``batch_size``).
    Variable-length sequences are right-padded with zeros, and
    ``waveform_lengths`` stores the original (unpadded) sample counts.

    Attributes
    ----------
    waveforms:
        Padded waveform tensor of shape ``(B, C, T_max)``.
    waveform_lengths:
        Original lengths in samples, shape ``(B,)``.
    sample_rate:
        Common sample rate of all waveforms in the batch.
    metadata:
        Per-sample metadata dicts (length ``B``). Not padded.
    attention_mask:
        Boolean mask of shape ``(B, T_max)`` where ``True`` indicates a
        valid (non-padding) sample.
    labels:
        Optional integer label tensor, shape ``(B, max_segments)``.
    label_mask:
        Optional boolean tensor indicating valid label positions.
    segments:
        Optional raw segment lists per sample (length ``B``).
    """

    waveforms: torch.Tensor                        # (B, C, T_max)
    waveform_lengths: torch.Tensor                  # (B,) dtype=long
    sample_rate: int
    metadata: list[MetadataDict]                    # len B
    attention_mask: torch.Tensor                    # (B, T_max) dtype=bool

    # Optional fields, populated by specific collators.
    labels: torch.Tensor | None = None              # (B, max_segments) dtype=long
    label_mask: torch.Tensor | None = None          # (B, max_segments) dtype=bool
    frame_labels: torch.Tensor | None = None        # (B, T_frames, n_labels) dtype=bool
    segments: list[SegmentList] | None = None       # len B

    @property
    def batch_size(self) -> int:
        return self.waveforms.shape[0]

    @property
    def max_length(self) -> int:
        """Maximum waveform length in samples (padded dimension)."""
        return self.waveforms.shape[-1]

    def to(self, device: torch.device | str) -> DataBatch:
        """Move all tensors to *device* and return ``self``."""
        self.waveforms = self.waveforms.to(device)
        self.waveform_lengths = self.waveform_lengths.to(device)
        self.attention_mask = self.attention_mask.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.label_mask is not None:
            self.label_mask = self.label_mask.to(device)
        if self.frame_labels is not None:
            self.frame_labels = self.frame_labels.to(device)
        return self

    def pin_memory(self) -> DataBatch:
        """Pin all tensors for faster host-to-device transfer.

        No-op if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return self
        self.waveforms = self.waveforms.pin_memory()
        self.waveform_lengths = self.waveform_lengths.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        if self.label_mask is not None:
            self.label_mask = self.label_mask.pin_memory()
        if self.frame_labels is not None:
            self.frame_labels = self.frame_labels.pin_memory()
        return self

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self) -> str:
        return (
            f"DataBatch(batch_size={self.batch_size}, "
            f"max_length={self.max_length}, "
            f"sample_rate={self.sample_rate})"
        )
