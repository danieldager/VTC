"""Abstract base class for offline feature processors.

A :class:`FeatureProcessor` wraps a model or tool that reads raw audio and
produces metadata (e.g. VAD timestamps, speaker labels, SNR estimates).

Implementations are responsible for their own parallelization strategy —
GPU batching, CPU multiprocessing, or SLURM array distribution are all valid
approaches.

Example
-------
::

    class VADProcessor(FeatureProcessor):
        name = "vad"
        version = "1.0.0"

        def process(self, wav_id, audio_path):
            segments = run_tenvad(audio_path)
            return {"segments": segments, "speech_ratio": compute_ratio(segments)}

        def save(self, wav_id, metadata, output_dir):
            path = output_dir / f"{wav_id}.parquet"
            write_segments(metadata["segments"], path)
            return path

        def load(self, wav_id, output_dir):
            return read_segments(output_dir / f"{wav_id}.parquet")

        def exists(self, wav_id, output_dir):
            return (output_dir / f"{wav_id}.parquet").is_file()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from dataloader.types import MetadataDict, WavID


class FeatureProcessor(ABC):
    """Base class for offline feature extraction.

    Each processor has a unique :attr:`name` used as the key in manifest
    joins, and a :attr:`version` for cache invalidation.

    Subclasses must implement:
    - :meth:`process` — run extraction on a single file
    - :meth:`save` — persist results to disk
    - :meth:`load` — read results back from disk
    - :meth:`exists` — check whether results already exist
    """

    # ── Identity ──────────────────────────────────────────────────────────
    name: ClassVar[str]
    """Short lowercase identifier (e.g. ``"vad"``, ``"vtc"``, ``"snr"``)."""

    version: ClassVar[str]
    """Semver string for cache‑busting (e.g. ``"1.0.0"``)."""

    # ── Core interface ────────────────────────────────────────────────────
    @abstractmethod
    def process(self, wav_id: WavID, audio_path: Path) -> MetadataDict:
        """Extract metadata from a single audio file.

        Parameters
        ----------
        wav_id:
            Unique identifier for this waveform.
        audio_path:
            Absolute path to the audio file on disk.

        Returns
        -------
        MetadataDict
            Extracted metadata. Schema is processor-specific but must be
            JSON-serializable for manifest compatibility.
        """

    @abstractmethod
    def save(self, wav_id: WavID, metadata: MetadataDict, output_dir: Path) -> Path:
        """Persist extracted metadata to *output_dir*.

        Parameters
        ----------
        wav_id:
            Waveform identifier.
        metadata:
            The dict returned by :meth:`process`.
        output_dir:
            Root directory for this processor's outputs.

        Returns
        -------
        Path
            The file (or directory) that was written.
        """

    @abstractmethod
    def load(self, wav_id: WavID, output_dir: Path) -> MetadataDict:
        """Load previously saved metadata.

        Parameters
        ----------
        wav_id:
            Waveform identifier.
        output_dir:
            Root directory for this processor's outputs.

        Returns
        -------
        MetadataDict
            The same dict that was passed to :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If no saved metadata exists for *wav_id*.
        """

    @abstractmethod
    def exists(self, wav_id: WavID, output_dir: Path) -> bool:
        """Check whether saved metadata exists for *wav_id*.

        This is used for checkpointing: skip files that have already been
        processed successfully.
        """

    # ── Optional hooks ────────────────────────────────────────────────────
    def validate(self, metadata: MetadataDict) -> bool:
        """Validate that *metadata* conforms to the expected schema.

        The default implementation always returns ``True``. Override to add
        schema checks.
        """
        return True

    # ── Dunder ────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, version={self.version!r})"
