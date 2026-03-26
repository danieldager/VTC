"""Typed manifest wrapper around Polars DataFrames.

A :class:`MetadataManifest` enforces a minimum schema contract: every
manifest has a ``wav_id`` column that serves as the join key across
heterogeneous metadata sources. Beyond that, columns are processor-specific.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from dataloader.types import WavID

log = logging.getLogger(__name__)

_WAV_ID_COL = "wav_id"


class MetadataManifest:
    """Typed manifest of metadata entries keyed by ``wav_id``.

    Parameters
    ----------
    df:
        Underlying Polars DataFrame. Must contain a ``wav_id`` column.
    source:
        Human-readable provenance tag (e.g. ``"vad_merged"``, ``"snr"``).

    Raises
    ------
    ValueError
        If *df* does not contain a ``wav_id`` column.
    """

    __slots__ = ("_df", "_source")

    def __init__(self, df: pl.DataFrame, source: str) -> None:
        if _WAV_ID_COL not in df.columns:
            raise ValueError(
                f"DataFrame must contain a {_WAV_ID_COL!r} column. "
                f"Got columns: {df.columns}"
            )
        # Ensure wav_id is always Utf8 (string).
        if df.schema[_WAV_ID_COL] != pl.Utf8:
            df = df.with_columns(pl.col(_WAV_ID_COL).cast(pl.Utf8))
        self._df = df
        self._source = source

    # ── Constructors ──────────────────────────────────────────────────────
    @classmethod
    def from_parquet(cls, path: Path | str, source: str | None = None) -> MetadataManifest:
        """Load manifest from a Parquet file.

        Parameters
        ----------
        path:
            Path to the ``.parquet`` file.
        source:
            Provenance tag. Defaults to the file stem.
        """
        path = Path(path)
        df = pl.read_parquet(path)
        return cls(df, source=source or path.stem)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, source: str) -> MetadataManifest:
        """Wrap an existing Polars DataFrame."""
        return cls(df, source=source)

    # ── Serialization ─────────────────────────────────────────────────────
    def to_parquet(self, path: Path | str) -> None:
        """Write the manifest to a Parquet file with zstd compression."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.write_parquet(path, compression="zstd")
        log.info("Wrote %d rows to %s", len(self._df), path)

    # ── DataFrame operations ──────────────────────────────────────────────
    def filter(self, predicate: pl.Expr) -> MetadataManifest:
        """Return a new manifest containing only rows matching *predicate*."""
        return MetadataManifest(self._df.filter(predicate), source=self._source)

    def select(self, columns: list[str]) -> MetadataManifest:
        """Return a new manifest with only the specified columns.

        ``wav_id`` is always included regardless of *columns*.
        """
        cols = [_WAV_ID_COL] + [c for c in columns if c != _WAV_ID_COL]
        return MetadataManifest(self._df.select(cols), source=self._source)

    def join(
        self,
        other: MetadataManifest,
        on: str = _WAV_ID_COL,
        how: str = "inner",
        *,
        suffix: str = "_right",
    ) -> MetadataManifest:
        """Join two manifests on *on* column.

        Parameters
        ----------
        other:
            The right manifest.
        on:
            Join key column name.
        how:
            Join strategy (``"inner"``, ``"left"``, ``"outer"``).
        suffix:
            Suffix for duplicate column names from *other*.
        """
        # Polars uses "full" for outer joins; coalesce merges the join key.
        polars_how: str = "full" if how == "outer" else how
        merged_df = self._df.join(
            other._df,
            on=on,
            how=polars_how,  # type: ignore[arg-type]
            suffix=suffix,
            coalesce=True,
        )
        combined_source = f"{self._source}+{other._source}"
        return MetadataManifest(merged_df, source=combined_source)

    def rename_columns(self, mapping: dict[str, str]) -> MetadataManifest:
        """Return a new manifest with renamed columns.

        ``wav_id`` cannot be renamed.
        """
        if _WAV_ID_COL in mapping:
            raise ValueError(f"Cannot rename the {_WAV_ID_COL!r} column.")
        return MetadataManifest(self._df.rename(mapping), source=self._source)

    # ── Accessors ─────────────────────────────────────────────────────────
    @property
    def wav_ids(self) -> list[WavID]:
        """Return all unique ``wav_id`` values."""
        return self._df[_WAV_ID_COL].unique().sort().to_list()

    @property
    def df(self) -> pl.DataFrame:
        """Return the underlying Polars DataFrame (read-only view)."""
        return self._df

    @property
    def source(self) -> str:
        """Provenance tag."""
        return self._source

    @property
    def columns(self) -> list[str]:
        """Column names."""
        return self._df.columns

    @property
    def schema(self) -> dict[str, pl.DataType]:
        """Column name → Polars dtype mapping."""
        return dict(self._df.schema)

    # ── Dunder ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        cols = ", ".join(self._df.columns[:6])
        if len(self._df.columns) > 6:
            cols += ", ..."
        return (
            f"MetadataManifest(source={self._source!r}, "
            f"rows={len(self._df)}, columns=[{cols}])"
        )

    def __contains__(self, wav_id: WavID) -> bool:
        return (
            self._df.filter(pl.col(_WAV_ID_COL) == wav_id).height > 0
        )
