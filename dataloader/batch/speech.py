"""Speech-specific collator with masking support.

:class:`SpeechCollator` pads waveforms to a uniform length, constructs
attention masks, and optionally collates frame-level label masks generated
by :class:`~dataloader.transform.label.MaskGenerator`.
"""

from __future__ import annotations

import torch

from dataloader.batch.base import Collator
from dataloader.batch.data_batch import DataBatch
from dataloader.types import MetadataDict, SampleRate, Waveform


class SpeechCollator(Collator):
    """Pad and collate speech samples into a :class:`DataBatch`.

    Parameters
    ----------
    pad_to_multiple_of:
        If set, pad waveforms so that the time dimension is a multiple
        of this value. Useful for models that require power-of-2 inputs
        or fixed chunk sizes.
    attention_mask_key:
        Metadata key for a precomputed frame-level attention mask
        (from :class:`MaskGenerator`). If present, it is collated into
        ``DataBatch.frame_labels``.
    label_mask_key:
        Metadata key for a precomputed frame-level label mask.
    """

    def __init__(
        self,
        pad_to_multiple_of: int | None = None,
        attention_mask_key: str = "attention_mask",
        label_mask_key: str = "label_mask",
    ) -> None:
        self._pad_multiple = pad_to_multiple_of
        self._attention_mask_key = attention_mask_key
        self._label_mask_key = label_mask_key

    def __call__(
        self,
        samples: list[tuple[Waveform, SampleRate, MetadataDict]],
    ) -> DataBatch:
        waveforms: list[Waveform] = []
        lengths: list[int] = []
        all_metadata: list[MetadataDict] = []
        sample_rate: int | None = None

        for wav, sr, meta in samples:
            if sample_rate is None:
                sample_rate = sr
            waveforms.append(wav)
            lengths.append(wav.shape[-1])
            all_metadata.append(meta)

        assert sample_rate is not None, "Cannot collate an empty batch."

        max_len = max(lengths)
        if self._pad_multiple:
            remainder = max_len % self._pad_multiple
            if remainder != 0:
                max_len += self._pad_multiple - remainder

        # Pad waveforms.
        channels = waveforms[0].shape[0]
        padded = torch.zeros(len(waveforms), channels, max_len, dtype=torch.float32)
        attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.bool)

        for i, (wav, length) in enumerate(zip(waveforms, lengths)):
            padded[i, :, :length] = wav
            attention_mask[i, :length] = True

        waveform_lengths = torch.tensor(lengths, dtype=torch.long)

        # Collate optional frame-level masks from metadata.
        frame_labels = self._collate_frame_masks(
            all_metadata, self._label_mask_key,
        )

        return DataBatch(
            waveforms=padded,
            waveform_lengths=waveform_lengths,
            sample_rate=sample_rate,
            metadata=all_metadata,
            attention_mask=attention_mask,
            frame_labels=frame_labels,
        )

    def _collate_frame_masks(
        self,
        metadata_list: list[MetadataDict],
        key: str,
    ) -> torch.Tensor | None:
        """Pad and stack frame-level boolean masks from metadata."""
        masks = [m.get(key) for m in metadata_list]
        if all(m is None for m in masks):
            return None

        # All masks must be tensors with same number of label dimensions.
        tensors = [m for m in masks if isinstance(m, torch.Tensor)]
        if not tensors:
            return None

        max_frames = max(t.shape[0] for t in tensors)
        n_labels = tensors[0].shape[-1] if tensors[0].dim() > 1 else 1
        padded = torch.zeros(len(metadata_list), max_frames, n_labels, dtype=torch.bool)

        for i, m in enumerate(masks):
            if isinstance(m, torch.Tensor):
                t = m if m.dim() > 1 else m.unsqueeze(-1)
                padded[i, : t.shape[0], : t.shape[1]] = t

        return padded

    def __repr__(self) -> str:
        return f"SpeechCollator(pad_to_multiple_of={self._pad_multiple})"
