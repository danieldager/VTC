"""Abstract base class for runtime data transforms.

:class:`DataProcessor` defines the contract for transforms that operate on
``(waveform, sample_rate, metadata)`` triples at training time. Transforms
are composable via :class:`Compose`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dataloader.types import MetadataDict, SampleRate, Waveform


class DataProcessor(ABC):
    """Base class for runtime transforms on waveform + metadata.

    Each processor receives a tuple of ``(waveform, sample_rate, metadata)``
    and returns a (possibly modified) version of the same triple. Processors
    may modify the waveform (e.g. resample, segment), the metadata (e.g.
    encode labels), or both.
    """

    @abstractmethod
    def __call__(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        """Apply the transform.

        Parameters
        ----------
        waveform:
            Audio tensor of shape ``(channels, samples)``.
        sample_rate:
            Sample rate in Hz.
        metadata:
            Metadata dict (may be modified in-place or replaced).

        Returns
        -------
        tuple[Waveform, SampleRate, MetadataDict]
            Transformed triple.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Compose(DataProcessor):
    """Chain multiple :class:`DataProcessor` transforms sequentially.

    Parameters
    ----------
    processors:
        Ordered list of transforms to apply. Each transform's output
        is fed as input to the next.

    Example
    -------
    ::

        pipeline = Compose([
            Resampler(target_sr=16_000),
            VADSegmenter(padding_s=0.1),
            LabelEncoder(labels=["FEM", "MAL", "KCHI", "OCH"]),
        ])

        waveform, sr, metadata = pipeline(waveform, sr, metadata)
    """

    def __init__(self, processors: list[DataProcessor]) -> None:
        self._processors = list(processors)

    def __call__(
        self,
        waveform: Waveform,
        sample_rate: SampleRate,
        metadata: MetadataDict,
    ) -> tuple[Waveform, SampleRate, MetadataDict]:
        for proc in self._processors:
            waveform, sample_rate, metadata = proc(waveform, sample_rate, metadata)
        return waveform, sample_rate, metadata

    def __repr__(self) -> str:
        procs = ", ".join(repr(p) for p in self._processors)
        return f"Compose([{procs}])"

    def __len__(self) -> int:
        return len(self._processors)

    def __getitem__(self, index: int) -> DataProcessor:
        return self._processors[index]
