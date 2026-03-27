"""Feature loaders — load pipeline outputs and attach them to clips.

Each loader knows how to:
1. **Detect** whether its feature data exists on disk.
2. **Load** file-level data for a given uid.
3. **Attach** per-clip slices of that data to a Clip's features dict.

The ``FeatureLoader`` protocol defines the interface.  Concrete loaders
(``VADLoader``, ``VTCLoader``, ``ESCLoader``) wrap the I/O that was
previously inline in ``package.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

from src.packaging.clips import Clip, Segment


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureLoader(Protocol):
    """Interface for loading a single kind of extracted feature."""

    name: str
    """Short identifier used as key prefix in ``Clip.features``."""

    def is_available(self, output_dir: Path) -> bool:
        """Return True if this feature's outputs exist on disk."""
        ...

    def load_file(self, output_dir: Path, uid: str) -> Any:
        """Load file-level feature data from disk for *uid*."""
        ...

    def attach_to_clip(self, clip: Clip, file_data: Any) -> None:
        """Slice *file_data* for this clip and store in ``clip.features``."""
        ...


# ---------------------------------------------------------------------------
# VTC loader
# ---------------------------------------------------------------------------


class VTCLoader:
    """Load merged VTC segments (speaker-labelled speech) from parquets."""

    name = "vtc"

    def is_available(self, output_dir: Path) -> bool:
        vtc_dir = output_dir / "vtc_merged"
        return vtc_dir.exists() and any(vtc_dir.glob("*.parquet"))

    def load_file(self, output_dir: Path, uid: str) -> list[Segment]:
        vtc_dir = output_dir / "vtc_merged"
        if not vtc_dir.exists():
            return []
        segments: list[Segment] = []
        for p in sorted(vtc_dir.glob("*.parquet")):
            df = pl.read_parquet(p).filter(pl.col("uid") == uid)
            for row in df.iter_rows(named=True):
                segments.append(
                    Segment(
                        onset=row["onset"],
                        offset=row["offset"],
                        label=row.get("label"),
                    )
                )
        return sorted(segments, key=lambda s: s.onset)

    def attach_to_clip(self, clip: Clip, file_data: list[Segment]) -> None:
        # Segments are already assigned by build_clips — nothing to do.
        pass


# ---------------------------------------------------------------------------
# VAD loader
# ---------------------------------------------------------------------------


class VADLoader:
    """Load merged VAD segments from a single parquet file."""

    name = "vad"

    def is_available(self, output_dir: Path) -> bool:
        return (output_dir / "vad_merged" / "segments.parquet").exists()

    def load_file(self, output_dir: Path, uid: str) -> list[Segment]:
        seg_path = output_dir / "vad_merged" / "segments.parquet"
        if not seg_path.exists():
            return []
        df = pl.read_parquet(seg_path).filter(pl.col("uid") == uid)
        return [
            Segment(onset=row["onset"], offset=row["offset"])
            for row in df.iter_rows(named=True)
        ]

    def attach_to_clip(self, clip: Clip, file_data: list[Segment]) -> None:
        # Segments are already assigned by build_clips — nothing to do.
        pass


# ---------------------------------------------------------------------------
# Noise loader (PANNs)
# ---------------------------------------------------------------------------


class ESCLoader:
    """Load PANNs ESC (Environmental Sound Classification) arrays and slice per clip."""

    name = "esc"

    def is_available(self, output_dir: Path) -> bool:
        esc_dir = output_dir / "esc"
        return esc_dir.exists() and any(esc_dir.glob("*.npz"))

    def load_file(
        self, output_dir: Path, uid: str
    ) -> tuple[np.ndarray | None, list[str], float]:
        """Return (categories_array, category_names, pool_step_s)."""
        esc_path = output_dir / "esc" / f"{uid}.npz"
        if not esc_path.exists():
            return None, [], 1.0
        data = np.load(esc_path, allow_pickle=True)
        cats = data["categories"].astype(np.float32)
        cat_names = list(data["category_names"])
        pool_step_s = float(data["pool_step_s"])
        return cats, cat_names, pool_step_s

    def attach_to_clip(
        self,
        clip: Clip,
        file_data: tuple[np.ndarray | None, list[str], float],
    ) -> None:
        file_esc, cat_names, pool_step_s = file_data
        if file_esc is None:
            return
        start_idx = int(clip.abs_onset / pool_step_s)
        end_idx = int(np.ceil(clip.abs_offset / pool_step_s))
        start_idx = max(0, min(start_idx, file_esc.shape[0]))
        end_idx = max(start_idx, min(end_idx, file_esc.shape[0]))
        clip.esc_array = file_esc[start_idx:end_idx].astype(np.float16)
        clip.esc_categories = cat_names
        clip.esc_step_s = pool_step_s


# ---------------------------------------------------------------------------
# Duration helper (VAD metadata)
# ---------------------------------------------------------------------------


def get_file_duration(output_dir: Path, uid: str) -> float | None:
    """Get audio duration from VAD metadata parquet."""
    meta_path = output_dir / "vad_meta" / "metadata.parquet"
    if not meta_path.exists():
        return None
    df = pl.read_parquet(meta_path).filter(pl.col("file_id") == uid)
    if df.is_empty():
        return None
    return float(df["duration"][0])
