"""Audio transforms: resampling, segmentation, normalization.

These are :class:`~dataloader.transform.base.DataProcessor` implementations
that modify the waveform tensor and adjust metadata accordingly.
"""

from __future__ import annotations

import torch

from dataloader.transform.base import DataProcessor
from dataloader.types import MetadataDict, SampleRate, SegmentList, Waveform


class Resampler(DataProcessor):
    """Resample waveform to a target sample rate.

    Parameters
    ----------
    target_sr:
        Desired sample rate in Hz.
    """

    def __init__(self, target_sr: SampleRate) -> None:
        self._target_sr = target_sr

    def __call__(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        if sample_rate == self._target_sr:
            return waveform, sample_rate, metadata
        try:
            import torchaudio.functional as F

            waveform = F.resample(waveform, orig_freq=sample_rate, new_freq=self._target_sr)
        except ImportError:
            import numpy as np
            from scipy.signal import resample as scipy_resample

            n_samples = int(waveform.shape[-1] * self._target_sr / sample_rate)
            resampled = np.asarray(
                scipy_resample(waveform.numpy(), n_samples, axis=-1),
                dtype=np.float32,
            )
            waveform = torch.from_numpy(resampled)

        return waveform, self._target_sr, metadata

    def __repr__(self) -> str:
        return f"Resampler(target_sr={self._target_sr})"


class VADSegmenter(DataProcessor):
    """Segment waveform using VAD timestamps.

    Extracts the active speech regions from the waveform based on VAD
    segment timestamps in metadata, concatenating them into a single
    contiguous tensor. Segment metadata is updated to reflect the new
    relative timestamps.

    Parameters
    ----------
    segments_key:
        Metadata key containing the :data:`SegmentList`.
    padding_s:
        Seconds of padding to add around each segment (clamped to
        waveform boundaries).
    output_key:
        Metadata key to store the modified segment list with new
        relative timestamps.
    """

    def __init__(
        self,
        segments_key: str = "vad_segments",
        padding_s: float = 0.0,
        output_key: str = "vad_segments_relative",
    ) -> None:
        self._segments_key = segments_key
        self._padding_s = padding_s
        self._output_key = output_key

    def __call__(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        segments: SegmentList = metadata.get(self._segments_key, [])  # type: ignore[assignment]
        if not segments:
            return waveform, sample_rate, metadata

        n_samples = waveform.shape[-1]
        chunks: list[Waveform] = []
        new_segments: SegmentList = []
        cursor = 0.0

        for seg in segments:
            onset_s = max(0.0, seg["onset"] - self._padding_s)
            offset_s = min(n_samples / sample_rate, seg["offset"] + self._padding_s)
            start = int(onset_s * sample_rate)
            end = int(offset_s * sample_rate)
            chunk = waveform[..., start:end]
            duration = chunk.shape[-1] / sample_rate
            chunks.append(chunk)
            new_segments.append({
                "onset": cursor,
                "offset": cursor + duration,
                **{k: v for k, v in seg.items() if k not in ("onset", "offset")},
            })
            cursor += duration

        if chunks:
            waveform = torch.cat(chunks, dim=-1)
        metadata = {**metadata, self._output_key: new_segments}
        return waveform, sample_rate, metadata

    def __repr__(self) -> str:
        return (
            f"VADSegmenter(segments_key={self._segments_key!r}, "
            f"padding_s={self._padding_s})"
        )


class Normalizer(DataProcessor):
    """Normalize waveform amplitude.

    Parameters
    ----------
    target_db:
        Target RMS level in dB. If ``None``, normalize to [-1, 1] peak.
    """

    def __init__(self, target_db: float | None = None) -> None:
        self._target_db = target_db

    def __call__(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        if self._target_db is not None:
            rms = waveform.pow(2).mean().sqrt()
            if rms > 0:
                target_rms = 10 ** (self._target_db / 20)
                waveform = waveform * (target_rms / rms)
        else:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
        return waveform, sample_rate, metadata

    def __repr__(self) -> str:
        return f"Normalizer(target_db={self._target_db})"
