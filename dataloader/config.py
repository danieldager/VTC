"""Pipeline and filter configuration for Dataloader++.

Two configuration types serve different purposes:

**PipelineConfig** records the hyperparameters used during offline feature
extraction (VAD threshold, VTC model, packaging clip size, etc.). Changing
any field requires rerunning the affected SLURM pipeline stage. Each config
gets a content-addressable ``version`` hash and can be saved alongside
outputs for reproducibility.

**FilterConfig** specifies data-selection criteria applied at DataLoader
creation time. These operate on the already-computed joined manifest and
require NO reprocessing — just Polars column filters.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from dataloader.manifest.schema import MetadataManifest

log = logging.getLogger(__name__)

_CONFIG_FILENAME = "pipeline_config.json"


# ── Pipeline configuration (extraction hyperparameters) ───────────────────────


@dataclass(frozen=True)
class PipelineConfig:
    """Extraction hyperparameters — changing any requires rerunning the pipeline.

    This is a provenance record saved alongside pipeline outputs. It documents
    exactly which parameters produced the data, enabling:

    - **Reproducibility**: re-run with identical params.
    - **Versioning**: different param sets → different ``version`` hashes →
      different output directories, so results from multiple configs coexist.

    Parameters
    ----------
    vad_threshold:
        TenVAD speech probability cutoff.
    vad_hop_size:
        TenVAD frame hop size in samples.
    vtc_threshold:
        Sigmoid threshold for VTC speech detection.
    vtc_model_config:
        Path to the segma model config YAML (relative to repo root).
    vtc_checkpoint:
        Path to the segma model checkpoint.
    snr_model_path:
        Path to the Brouhaha model checkpoint.
    esc_pool_window:
        PANNs temporal pooling window in seconds.
    esc_inference_window:
        PANNs chunk length in seconds.
    max_clip_s:
        Maximum clip duration for packaging (seconds).
    split_search_s:
        Search window for finding silence-based split points (seconds).
    target_sr:
        Target sample rate for packaged audio (Hz).
    audio_fmt:
        Audio codec in WebDataset shards (``"wav"`` or ``"flac"``).
    shard_size:
        Maximum clips per tar shard.
    """

    # VAD
    vad_threshold: float = 0.5
    vad_hop_size: int = 256
    # VTC
    vtc_threshold: float = 0.5
    vtc_model_config: str = ""
    vtc_checkpoint: str = ""
    # SNR (Brouhaha)
    snr_model_path: str = ""
    # ESC (PANNs)
    esc_pool_window: float = 1.0
    esc_inference_window: float = 10.0
    # Packaging
    max_clip_s: float = 600.0
    split_search_s: float = 120.0
    target_sr: int = 16_000
    audio_fmt: str = "wav"
    shard_size: int = 100

    # ── Versioning ────────────────────────────────────────────────────────

    @property
    def version(self) -> str:
        """Short content-addressable hash of all extraction parameters.

        Two configs with identical params produce the same hash. Useful for
        naming output directories: ``output/{dataset}/{config.version}/``.
        """
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Write config to a JSON file.

        If *path* is a directory, writes to ``{path}/pipeline_config.json``.
        """
        path = Path(path)
        if path.is_dir():
            path = path / _CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        log.info("Saved PipelineConfig (version=%s) to %s", self.version, path)

    @classmethod
    def load(cls, path: Path | str) -> PipelineConfig:
        """Load config from a JSON file or directory containing one."""
        path = Path(path)
        if path.is_dir():
            path = path / _CONFIG_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No pipeline config at {path}")
        data = json.loads(path.read_text())
        # Only pass known fields (forward-compatible with future additions).
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __repr__(self) -> str:
        return f"PipelineConfig(version={self.version!r})"


# ── Filter configuration (load-time data selection) ───────────────────────────


@dataclass
class FilterConfig:
    """Load-time data selection — applied to the joined manifest, no reprocessing.

    Every field is optional. ``None`` means "no filter on this criterion."
    Filters are AND-combined: a row must satisfy ALL non-None conditions.

    Parameters
    ----------
    min_duration_s / max_duration_s:
        File duration bounds (seconds).
    min_speech_ratio / max_speech_ratio:
        VAD speech ratio bounds (0–1).
    min_snr_db / max_snr_db:
        Mean SNR bounds (dB).
    min_c50_db / max_c50_db:
        Mean C50 clarity bounds (dB).
    min_vtc_segments:
        Minimum number of VTC segments.
    required_labels:
        VTC ``vtc_label_counts`` must contain at least one of these labels.
    excluded_esc_categories:
        Exclude files where ``dominant_category`` is in this list.
    max_dominant_esc_prob:
        Exclude files where the dominant ESC probability exceeds this.
    """

    # Duration
    min_duration_s: float | None = None
    max_duration_s: float | None = None
    # Speech density (VAD)
    min_speech_ratio: float | None = None
    max_speech_ratio: float | None = None
    # Acoustic quality (Brouhaha)
    min_snr_db: float | None = None
    max_snr_db: float | None = None
    min_c50_db: float | None = None
    max_c50_db: float | None = None
    # Speaker content (VTC)
    min_vtc_segments: int | None = None
    required_labels: list[str] | None = field(default=None)
    # ESC environment
    excluded_esc_categories: list[str] | None = field(default=None)
    max_dominant_esc_prob: float | None = None

    def apply(self, manifest: MetadataManifest) -> MetadataManifest:
        """Return a new manifest containing only rows that pass all filters.

        Parameters
        ----------
        manifest:
            A :class:`MetadataManifest` (typically the output of
            :meth:`ManifestJoiner.join`).

        Returns
        -------
        MetadataManifest
            Filtered manifest with the same schema but fewer rows.
        """
        df = manifest.df
        cols = set(df.columns)
        exprs: list[pl.Expr] = []

        # Duration filters
        if self.min_duration_s is not None and "duration" in cols:
            exprs.append(pl.col("duration") >= self.min_duration_s)
        if self.max_duration_s is not None and "duration" in cols:
            exprs.append(pl.col("duration") <= self.max_duration_s)

        # Speech ratio filters
        if self.min_speech_ratio is not None and "speech_ratio" in cols:
            exprs.append(pl.col("speech_ratio") >= self.min_speech_ratio)
        if self.max_speech_ratio is not None and "speech_ratio" in cols:
            exprs.append(pl.col("speech_ratio") <= self.max_speech_ratio)

        # SNR filters
        if self.min_snr_db is not None and "snr_mean" in cols:
            exprs.append(pl.col("snr_mean") >= self.min_snr_db)
        if self.max_snr_db is not None and "snr_mean" in cols:
            exprs.append(pl.col("snr_mean") <= self.max_snr_db)

        # C50 filters
        if self.min_c50_db is not None and "c50_mean" in cols:
            exprs.append(pl.col("c50_mean") >= self.min_c50_db)
        if self.max_c50_db is not None and "c50_mean" in cols:
            exprs.append(pl.col("c50_mean") <= self.max_c50_db)

        # VTC segment count
        if self.min_vtc_segments is not None and "vtc_n_segments" in cols:
            exprs.append(pl.col("vtc_n_segments") >= self.min_vtc_segments)

        # Required speaker labels (VTC label_counts is a JSON string)
        if self.required_labels and "vtc_label_counts" in cols:
            label_expr = pl.lit(False)
            for label in self.required_labels:
                label_expr = label_expr | pl.col("vtc_label_counts").str.contains(
                    f'"{label}"'
                )
            exprs.append(label_expr)

        # Excluded ESC categories
        if self.excluded_esc_categories and "dominant_category" in cols:
            exprs.append(
                ~pl.col("dominant_category").is_in(self.excluded_esc_categories)
            )

        # Dominant ESC probability cap
        if self.max_dominant_esc_prob is not None and "dominant_prob" in cols:
            exprs.append(pl.col("dominant_prob") <= self.max_dominant_esc_prob)

        # Combine all filters (AND)
        if not exprs:
            return manifest

        combined = exprs[0]
        for expr in exprs[1:]:
            combined = combined & expr

        before = len(df)
        filtered = manifest.filter(combined)
        after = len(filtered)
        log.info(
            "FilterConfig: %d → %d rows (%d excluded)",
            before,
            after,
            before - after,
        )
        return filtered

    def save(self, path: Path | str) -> None:
        """Write filter config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @property
    def is_active(self) -> bool:
        """True if any filter criterion is set."""
        return any(getattr(self, f.name) is not None for f in fields(self))

    @classmethod
    def load(cls, path: Path | str) -> FilterConfig:
        """Load filter config from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ── Loader configuration (DataLoader / streaming settings) ────────────────────


@dataclass
class LoaderConfig:
    """PyTorch DataLoader and streaming settings.

    These control how the WebDataset is served: batching, parallelism,
    shuffling, and audio decoding options.

    Parameters
    ----------
    batch_size:
        Samples per batch.
    num_workers:
        DataLoader worker processes.
    prefetch_factor:
        Batches prefetched per worker (None = PyTorch default).
    shuffle_buffer:
        WebDataset in-memory shuffle buffer size.
    audio_key:
        Extension key for audio in tar shards (``"flac"`` or ``"wav"``).
    metadata_keys:
        Extension keys for metadata in tar shards.
    seed:
        Random seed for shard and sample shuffling.
    pin_memory:
        Pin batch tensors into CUDA pinned memory.
    drop_last:
        Drop the last incomplete batch.
    pad_to_multiple_of:
        Pad waveforms so time dim is a multiple of this value.
    """

    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int | None = 2
    shuffle_buffer: int = 1000
    audio_key: str = "wav"
    metadata_keys: list[str] = field(default_factory=lambda: ["json"])
    seed: int = 42
    pin_memory: bool = True
    drop_last: bool = True
    pad_to_multiple_of: int | None = None


# ── Top-level dataset configuration ──────────────────────────────────────────

_DATASET_CONFIG_FILENAME = "dataset_config.json"


@dataclass
class DatasetConfig:
    """Unified configuration for end-to-end dataset → DataLoader creation.

    A single file that specifies everything needed to go from a processed
    dataset directory to a ready-to-train PyTorch DataLoader:

    - **dataset_dir**: path to the output directory containing pipeline
      outputs and packaged ``.tar`` shards.
    - **pipeline**: extraction hyperparameters (provenance record).
    - **filters**: load-time data selection criteria.
    - **loader**: DataLoader and streaming settings.

    Example config file::

        {
          "dataset_dir": "output/seedlings_1",
          "pipeline": {"vad_threshold": 0.5, "target_sr": 16000},
          "filters": {"min_snr_db": 10.0, "required_labels": ["KCHI"]},
          "loader": {"batch_size": 16, "num_workers": 4}
        }
    """

    dataset_dir: str
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)

    def save(self, path: Path | str) -> None:
        """Write the full config to a JSON file.

        If *path* is a directory, writes ``dataset_config.json`` inside it.
        """
        path = Path(path)
        if path.is_dir():
            path = path / _DATASET_CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        log.info("Saved DatasetConfig to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> DatasetConfig:
        """Load a full config from a JSON file or directory."""
        path = Path(path)
        if path.is_dir():
            path = path / _DATASET_CONFIG_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No dataset config at {path}")
        data = json.loads(path.read_text())

        pipeline_fields = {f.name for f in fields(PipelineConfig)}
        filter_fields = {f.name for f in fields(FilterConfig)}
        loader_fields = {f.name for f in fields(LoaderConfig)}

        return cls(
            dataset_dir=data["dataset_dir"],
            pipeline=PipelineConfig(
                **{k: v for k, v in data.get("pipeline", {}).items()
                   if k in pipeline_fields}
            ),
            filters=FilterConfig(
                **{k: v for k, v in data.get("filters", {}).items()
                   if k in filter_fields}
            ),
            loader=LoaderConfig(
                **{k: v for k, v in data.get("loader", {}).items()
                   if k in loader_fields}
            ),
        )

    def __repr__(self) -> str:
        return (
            f"DatasetConfig(dataset_dir={self.dataset_dir!r}, "
            f"pipeline={self.pipeline!r}, "
            f"filters_active={self.filters.is_active})"
        )
