"""Abstract base class for feature loaders.

A :class:`FeatureLoader` provides read access to both waveforms and their
associated metadata for a set of waveform IDs. It is the online counterpart
of :class:`~dataloader.processor.base.FeatureProcessor` — processors write
data, loaders read it.

The base class defines the contract; concrete implementations handle
format-specific I/O (WebDataset shards, raw files, HDF5, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dataloader.types import MetadataDict, SampleRate, WavID, Waveform


class FeatureLoader(ABC):
    """Base class for loading waveform and metadata for a single sample.

    Subclasses must implement:
    - :meth:`load_waveform` — decode audio to a tensor
    - :meth:`load_metadata` — read precomputed metadata
    - :meth:`available_ids` — enumerate loadable samples
    """

    @abstractmethod
    def load_waveform(self, wav_id: WavID) -> tuple[Waveform, SampleRate]:
        """Load and decode a waveform.

        Parameters
        ----------
        wav_id:
            Unique waveform identifier.

        Returns
        -------
        tuple[Waveform, SampleRate]
            A ``(channels, samples)`` float32 tensor and its sample rate.

        Raises
        ------
        FileNotFoundError
            If the audio file for *wav_id* cannot be found.
        """

    @abstractmethod
    def load_metadata(self, wav_id: WavID) -> MetadataDict:
        """Load precomputed metadata for a waveform.

        Parameters
        ----------
        wav_id:
            Unique waveform identifier.

        Returns
        -------
        MetadataDict
            Metadata dict whose schema depends on the pipeline stages
            that produced it.

        Raises
        ------
        FileNotFoundError
            If no metadata exists for *wav_id*.
        """

    def load(self, wav_id: WavID) -> tuple[Waveform, SampleRate, MetadataDict]:
        """Load both waveform and metadata in a single call.

        The default implementation calls :meth:`load_waveform` and
        :meth:`load_metadata` sequentially. Override if the storage format
        provides a more efficient combined read path (e.g. WebDataset tar
        seeks).
        """
        waveform, sample_rate = self.load_waveform(wav_id)
        metadata = self.load_metadata(wav_id)
        return waveform, sample_rate, metadata

    @abstractmethod
    def available_ids(self) -> list[WavID]:
        """Return all wav_ids that this loader can serve.

        The returned list should be deterministically ordered (e.g. sorted)
        for reproducibility.
        """

    def __contains__(self, wav_id: WavID) -> bool:
        """Check whether *wav_id* is available for loading."""
        return wav_id in self.available_ids()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
