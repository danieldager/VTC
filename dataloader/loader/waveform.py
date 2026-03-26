"""Waveform loading utilities.

:class:`WaveformLoader` handles decoding audio files from disk into
PyTorch tensors. It supports multiple backends (soundfile, torchaudio)
and optional resampling at load time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dataloader.types import SampleRate, WavID, Waveform

log = logging.getLogger(__name__)


class WaveformLoader:
    """Load audio files into tensors.

    Parameters
    ----------
    audio_dir:
        Root directory containing audio files. Files are resolved as
        ``{audio_dir}/{wav_id}.{extension}`` unless *path_map* is given.
    extension:
        Default audio file extension (without dot).
    target_sr:
        If set, resample all loaded audio to this sample rate.
    mono:
        If ``True``, downmix multi-channel audio to mono.
    path_map:
        Optional mapping from ``wav_id`` to absolute audio path.
        Overrides the ``{audio_dir}/{wav_id}.{extension}`` convention.
    """

    def __init__(
        self,
        audio_dir: Path | str | None = None,
        extension: str = "wav",
        target_sr: SampleRate | None = None,
        mono: bool = True,
        path_map: dict[WavID, Path | str] | None = None,
    ) -> None:
        self._audio_dir = Path(audio_dir) if audio_dir is not None else None
        self._extension = extension
        self._target_sr = target_sr
        self._mono = mono
        self._path_map = {k: Path(v) for k, v in path_map.items()} if path_map else {}

    def resolve_path(self, wav_id: WavID) -> Path:
        """Resolve the audio file path for *wav_id*.

        Raises
        ------
        FileNotFoundError
            If the resolved path does not exist.
        """
        if wav_id in self._path_map:
            path = self._path_map[wav_id]
        elif self._audio_dir is not None:
            path = self._audio_dir / f"{wav_id}.{self._extension}"
        else:
            raise FileNotFoundError(
                f"No audio_dir or path_map entry for wav_id={wav_id!r}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path

    def load(self, wav_id: WavID) -> tuple[Waveform, SampleRate]:
        """Load and decode an audio file.

        Returns
        -------
        tuple[Waveform, SampleRate]
            Float32 tensor of shape ``(channels, samples)`` and sample rate.
        """
        path = self.resolve_path(wav_id)
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # data shape: (samples, channels) — transpose to (channels, samples).
        waveform = torch.from_numpy(data.T)

        if self._mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self._target_sr is not None and sr != self._target_sr:
            waveform = self._resample(waveform, sr, self._target_sr)
            sr = self._target_sr

        return waveform, sr

    @staticmethod
    def _resample(
        waveform: Waveform,
        orig_sr: SampleRate,
        target_sr: SampleRate,
    ) -> Waveform:
        """Resample using torchaudio if available, else scipy."""
        try:
            import torchaudio.functional as F

            return F.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)
        except ImportError:
            from scipy.signal import resample as scipy_resample

            n_samples = int(waveform.shape[-1] * target_sr / orig_sr)
            resampled = np.asarray(
                scipy_resample(waveform.numpy(), n_samples, axis=-1),
                dtype=np.float32,
            )
            return torch.from_numpy(resampled)

    def __repr__(self) -> str:
        return (
            f"WaveformLoader(audio_dir={str(self._audio_dir)!r}, "
            f"extension={self._extension!r}, target_sr={self._target_sr})"
        )
