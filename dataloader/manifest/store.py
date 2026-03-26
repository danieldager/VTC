"""Unified metadata I/O abstraction.

:class:`MetadataStore` defines a format-agnostic interface for reading and
writing per-file metadata. Concrete backends handle Parquet, NPZ, and other
formats behind the same API, so that downstream code (loaders, joiners) does
not need to know how metadata is serialized.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import polars as pl

from dataloader.types import MetadataDict, WavID

log = logging.getLogger(__name__)


class MetadataStore(ABC):
    """Abstract interface for per-file metadata I/O.

    Parameters
    ----------
    root:
        Root directory containing all metadata files managed by this store.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @abstractmethod
    def load(self, wav_id: WavID) -> MetadataDict:
        """Load metadata for a single waveform.

        Raises
        ------
        FileNotFoundError
            If no metadata exists for *wav_id*.
        """

    @abstractmethod
    def save(self, wav_id: WavID, data: MetadataDict) -> None:
        """Persist metadata for a single waveform."""

    @abstractmethod
    def exists(self, wav_id: WavID) -> bool:
        """Check whether metadata for *wav_id* has been saved."""

    @abstractmethod
    def list_ids(self) -> list[WavID]:
        """Return all wav_ids that have stored metadata."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={str(self._root)!r})"


# ── Concrete backends ─────────────────────────────────────────────────────────


class ParquetStore(MetadataStore):
    """Metadata stored as rows in Parquet files.

    Supports two modes:

    1. **Single-file**: All metadata in one ``{root}/metadata.parquet`` file,
       filtered by ``wav_id`` column.
    2. **Sharded**: Multiple ``{root}/shard_*.parquet`` files, lazily scanned
       and concatenated.

    Parameters
    ----------
    root:
        Directory containing ``.parquet`` file(s).
    wav_id_column:
        Column name used as the waveform identifier (default ``"uid"``
        for backward compatibility with existing VTC pipeline outputs).
    """

    def __init__(
        self,
        root: Path | str,
        wav_id_column: str = "uid",
    ) -> None:
        super().__init__(root)
        self._wav_id_col = wav_id_column
        self._cache: pl.DataFrame | None = None

    def _load_all(self) -> pl.DataFrame:
        """Lazily load and cache all Parquet files under root."""
        if self._cache is not None:
            return self._cache

        pq_files = sorted(self._root.glob("*.parquet"))
        if not pq_files:
            raise FileNotFoundError(
                f"No .parquet files found in {self._root}"
            )

        frames = [pl.read_parquet(f) for f in pq_files]
        self._cache = pl.concat(frames, how="diagonal_relaxed")
        log.debug(
            "Loaded %d rows from %d parquet file(s) in %s",
            len(self._cache),
            len(pq_files),
            self._root,
        )
        return self._cache

    def load(self, wav_id: WavID) -> MetadataDict:
        df = self._load_all()
        rows = df.filter(pl.col(self._wav_id_col) == wav_id)
        if rows.is_empty():
            raise FileNotFoundError(
                f"No metadata for wav_id={wav_id!r} in {self._root}"
            )
        # Return as list of dicts (one per segment/row).
        return {"rows": rows.to_dicts(), "wav_id": wav_id}

    def save(self, wav_id: WavID, data: MetadataDict) -> None:
        # For Parquet stores, saving appends or overwrites a shard.
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{wav_id}.parquet"
        if "rows" in data and isinstance(data["rows"], list):
            df = pl.DataFrame(data["rows"])
        else:
            df = pl.DataFrame([data])
        df.write_parquet(path, compression="zstd")
        self._cache = None  # Invalidate cache.

    def exists(self, wav_id: WavID) -> bool:
        try:
            df = self._load_all()
        except FileNotFoundError:
            return False
        return df.filter(pl.col(self._wav_id_col) == wav_id).height > 0

    def list_ids(self) -> list[WavID]:
        try:
            df = self._load_all()
        except FileNotFoundError:
            return []
        return df[self._wav_id_col].unique().sort().to_list()


class NpzStore(MetadataStore):
    """Metadata stored as per-file ``.npz`` archives.

    Each waveform's metadata is a single ``.npz`` file at
    ``{root}/{wav_id}.npz``. Keys within the archive map to numpy arrays.

    This matches the existing SNR and Noise pipeline output format.
    """

    def load(self, wav_id: WavID) -> MetadataDict:
        path = self._root / f"{wav_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"No NPZ file at {path}")
        with np.load(path, allow_pickle=False) as npz:
            return {key: npz[key] for key in npz.files}

    def save(self, wav_id: WavID, data: MetadataDict) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{wav_id}.npz"
        arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]

    def exists(self, wav_id: WavID) -> bool:
        return (self._root / f"{wav_id}.npz").is_file()

    def list_ids(self) -> list[WavID]:
        return sorted(p.stem for p in self._root.glob("*.npz"))


class JsonStore(MetadataStore):
    """Metadata stored as per-file ``.json`` files.

    Each waveform's metadata is a single ``.json`` file at
    ``{root}/{wav_id}.json``.
    """

    def load(self, wav_id: WavID) -> MetadataDict:
        path = self._root / f"{wav_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No JSON file at {path}")
        with open(path) as f:
            return json.load(f)

    def save(self, wav_id: WavID, data: MetadataDict) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{wav_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def exists(self, wav_id: WavID) -> bool:
        return (self._root / f"{wav_id}.json").is_file()

    def list_ids(self) -> list[WavID]:
        return sorted(p.stem for p in self._root.glob("*.json"))
