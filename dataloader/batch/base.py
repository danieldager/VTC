"""Abstract base class for collators.

A :class:`Collator` takes a list of individual samples produced by a
:class:`~dataloader.dataset.base.SpeechDataset` and combines them into
a :class:`~dataloader.batch.data_batch.DataBatch` suitable for model input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dataloader.batch.data_batch import DataBatch
from dataloader.types import MetadataDict, SampleRate, Waveform


class Collator(ABC):
    """Base class for batching and collation.

    Implementations handle padding, mask construction, and optional
    label encoding for a batch of heterogeneous-length samples.

    This class is designed to be passed as the ``collate_fn`` argument
    to ``torch.utils.data.DataLoader``::

        loader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=SpeechCollator(sample_rate=16_000),
        )
    """

    @abstractmethod
    def __call__(
        self,
        samples: list[tuple[Waveform, SampleRate, MetadataDict]],
    ) -> DataBatch:
        """Collate a list of samples into a batch.

        Parameters
        ----------
        samples:
            List of ``(waveform, sample_rate, metadata)`` tuples as
            returned by a :class:`SpeechDataset`.

        Returns
        -------
        DataBatch
            Collated batch with padded tensors and masks.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
