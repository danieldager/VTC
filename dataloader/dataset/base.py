"""Abstract base class for speech datasets.

:class:`SpeechDataset` bridges a :class:`~dataloader.loader.base.FeatureLoader`
with an optional :class:`~dataloader.transform.base.DataProcessor` into
a standard ``torch.utils.data.Dataset``. Concrete subclasses implement
index mapping (random access vs. iterable/streaming).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from dataloader.types import MetadataDict, SampleRate, WavID, Waveform

if TYPE_CHECKING:
    from dataloader.loader.base import FeatureLoader
    from dataloader.transform.base import DataProcessor


class SpeechDataset(Dataset, ABC):
    """Base class for speech datasets.

    Combines a :class:`FeatureLoader` (I/O) with an optional
    :class:`DataProcessor` (runtime transforms) into a standard
    PyTorch :class:`Dataset`.

    Parameters
    ----------
    loader:
        Feature loader providing waveform and metadata I/O.
    processor:
        Optional runtime transform pipeline applied to each sample.
    """

    def __init__(
        self,
        loader: FeatureLoader,
        processor: DataProcessor | None = None,
    ) -> None:
        self._loader = loader
        self._processor = processor

    @property
    def loader(self) -> FeatureLoader:
        return self._loader

    @property
    def processor(self) -> DataProcessor | None:
        return self._processor

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        """Load and transform a single sample by index.

        Parameters
        ----------
        index:
            Integer index in ``[0, len(self))``.

        Returns
        -------
        tuple[Waveform, SampleRate, MetadataDict]
            The (possibly transformed) waveform, sample rate, and metadata.
        """

    def _load_and_process(
        self, wav_id: WavID
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        """Load a sample and apply the processor pipeline."""
        waveform, sample_rate, metadata = self._loader.load(wav_id)
        if self._processor is not None:
            waveform, sample_rate, metadata = self._processor(
                waveform, sample_rate, metadata
            )
        return waveform, sample_rate, metadata

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(loader={self._loader!r}, "
            f"processor={self._processor!r})"
        )
