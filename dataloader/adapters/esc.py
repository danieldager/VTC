"""ESC adapter — exposes ``src/pipeline/esc`` outputs as a FeatureProcessor.

Output directory layout consumed by this adapter::

    {output_dir}/
        esc_meta/shard_*.parquet  uid, esc_status, dominant_category, …
        esc/{uid}.npz             categories, audioset_probs (float16 arrays)
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from dataloader.processor.base import FeatureProcessor
from dataloader.types import MetadataDict, WavID


class ESCAdapter(FeatureProcessor):
    """Read-only adapter over existing ESC (PANNs) pipeline outputs.

    The ESC pipeline (``src/pipeline/esc.py``) runs via SLURM and
    produces sharded metadata parquets + per-file ``.npz`` arrays.

    Parameters
    ----------
    output_dir:
        Root output directory for the dataset (e.g. ``output/seedlings_1``).
    """

    name: ClassVar[str] = "esc"
    version: ClassVar[str] = "1.0.0"

    def __init__(self, output_dir: Path | str) -> None:
        self._root = Path(output_dir)
        self._meta_cache: pl.DataFrame | None = None

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _meta_df(self) -> pl.DataFrame:
        if self._meta_cache is None:
            path = self._root / "esc_meta"
            pq_files = sorted(path.glob("shard_*.parquet"))
            if not pq_files:
                raise FileNotFoundError(f"No shard parquets in {path}")
            self._meta_cache = pl.concat(
                [pl.read_parquet(f) for f in pq_files],
                how="diagonal_relaxed",
            )
        return self._meta_cache

    def _npz_path(self, wav_id: WavID) -> Path:
        return self._root / "esc" / f"{wav_id}.npz"

    # ── FeatureProcessor interface ────────────────────────────────────────

    def process(self, wav_id: WavID, audio_path: Path) -> MetadataDict:
        """Not supported — run ``sbatch slurm/esc.slurm`` instead."""
        raise NotImplementedError(
            "ESCAdapter is read-only. Run the Noise pipeline via SLURM: "
            "sbatch slurm/esc.slurm"
        )

    def save(self, wav_id: WavID, metadata: MetadataDict, output_dir: Path) -> Path:
        """Not supported — the Noise pipeline writes its own outputs."""
        raise NotImplementedError(
            "ESCAdapter is read-only. The pipeline saves outputs directly."
        )

    def load(self, wav_id: WavID, output_dir: Path | None = None) -> MetadataDict:
        """Load ESC metadata and per-frame arrays for a single file.

        Returns
        -------
        MetadataDict
            Keys: ``meta`` (dict of summary stats), ``categories``
            (float16 array, n_bins x n_cats), ``category_names``
            (string array), ``audioset_probs`` (float16 array, n_bins x 527),
            ``pool_step_s`` (float).
        """
        meta_df = self._meta_df()
        meta_rows = meta_df.filter(pl.col("uid") == wav_id)
        if meta_rows.is_empty():
            raise FileNotFoundError(
                f"No ESC metadata for wav_id={wav_id!r}"
            )

        npz_path = self._npz_path(wav_id)
        if not npz_path.is_file():
            raise FileNotFoundError(f"No ESC arrays at {npz_path}")

        with np.load(npz_path, allow_pickle=False) as npz:
            arrays = {
                "categories": npz["categories"],
                "category_names": npz["category_names"],
                "audioset_probs": npz["audioset_probs"],
                "pool_step_s": float(npz["pool_step_s"]),
            }

        return {
            "wav_id": wav_id,
            "meta": meta_rows.row(0, named=True),
            **arrays,
        }

    def exists(self, wav_id: WavID, output_dir: Path | None = None) -> bool:
        """Check whether ESC outputs exist for *wav_id*."""
        try:
            meta_df = self._meta_df()
        except FileNotFoundError:
            return False
        has_meta = meta_df.filter(
            (pl.col("uid") == wav_id) & (pl.col("esc_status") == "ok")
        ).height > 0
        return has_meta and self._npz_path(wav_id).is_file()

    # ── Convenience ───────────────────────────────────────────────────────

    def list_ids(self) -> list[WavID]:
        """Return all successfully processed wav_ids."""
        try:
            df = self._meta_df()
        except FileNotFoundError:
            return []
        return (
            df.filter(pl.col("esc_status") == "ok")
            .get_column("uid")
            .unique()
            .sort()
            .to_list()
        )

    def as_manifest(self) -> pl.DataFrame:
        """Return the full metadata DataFrame with ``wav_id`` column."""
        return self._meta_df().rename({"uid": "wav_id"})
