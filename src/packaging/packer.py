"""Packer — build clips from pipeline outputs and write WebDataset shards.

The ``Packer`` class is the top-level orchestrator for the packaging step.
It discovers available features, builds clips, attaches per-clip feature
data via ``FeatureLoader`` instances, and writes tar shards.

Usage (programmatic)::

    from src.packaging.packer import Packer

    packer = Packer(output_dir, manifest_df, audio_paths)
    packer.run()                       # build clips + write shards
    packer.save_stats()                # optional: compute & cache DataFrames
    packer.save_figures(fig_dir)       # optional: render dashboard PNGs

Or from the CLI via ``python -m src.pipeline.package``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl

from src.packaging.clips import CUT_TIERS, Clip, Segment, build_clips
from src.packaging.loaders import (
    FeatureLoader,
    ESCLoader,
    VADLoader,
    VTCLoader,
    get_file_duration,
)
from src.packaging.writer import write_shards


class Packer:
    """Build audio clips and write them as WebDataset shards.

    Parameters
    ----------
    output_dir : Path
        Pipeline output directory (contains ``vtc_merged/``, ``vad_merged/``, etc.)
    manifest_df : pl.DataFrame
        Manifest with a ``path`` column pointing to audio files.
    max_clip_s : float
        Maximum clip duration in seconds.
    split_search_s : float
        Search window for finding split points.
    audio_fmt : str
        ``"flac"`` or ``"wav"``.
    shard_size : int
        Maximum clips per shard.
    target_sr : int
        Target sample rate for output audio.
    extra_loaders : list[FeatureLoader] | None
        Additional feature loaders beyond the built-in VAD/VTC/Noise.
    """

    def __init__(
        self,
        output_dir: Path,
        manifest_df: pl.DataFrame,
        *,
        max_clip_s: float = 600.0,
        split_search_s: float = 120.0,
        audio_fmt: str = "wav",
        shard_size: int = 100,
        target_sr: int = 16_000,
        extra_loaders: list[FeatureLoader] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.manifest_df = manifest_df
        self.max_clip_s = max_clip_s
        self.split_search_s = split_search_s
        self.audio_fmt = audio_fmt
        self.shard_size = shard_size
        self.target_sr = target_sr

        # Discover available feature loaders
        self._loaders: list[FeatureLoader] = []
        candidates: list[FeatureLoader] = [
            VTCLoader(),
            VADLoader(),
            ESCLoader(),
            *(extra_loaders or []),
        ]
        for loader in candidates:
            if loader.is_available(output_dir):
                self._loaders.append(loader)
                print(f"  feature: {loader.name}")
            else:
                print(f"  feature: {loader.name} (not available)")

        # Built after run()
        self.all_clips: list[tuple[str, Path, int, Clip]] = []
        self.tier_counts: dict[str, int] = {t: 0 for t in CUT_TIERS}
        self.skipped: int = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def has_vad(self) -> bool:
        return any(l.name == "vad" for l in self._loaders)

    @property
    def has_esc(self) -> bool:
        return any(l.name == "esc" for l in self._loaders)

    @property
    def loader_names(self) -> list[str]:
        return [l.name for l in self._loaders]

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def build_clips(self) -> list[tuple[str, Path, int, Clip]]:
        """Build clips for every file in the manifest.

        Populates ``self.all_clips``, ``self.tier_counts``, ``self.skipped``.
        Returns ``self.all_clips`` for convenience.
        """
        t0 = time.time()
        self.all_clips = []
        self.tier_counts = {t: 0 for t in CUT_TIERS}
        self.skipped = 0
        n_files = len(self.manifest_df)

        # Index loaders by name for quick lookup
        loaders_by_name: dict[str, FeatureLoader] = {
            l.name: l for l in self._loaders
        }

        for row in self.manifest_df.iter_rows(named=True):
            audio_path = Path(row["path"])
            uid = audio_path.stem

            # Load file-level features
            file_features: dict[str, Any] = {}
            for loader in self._loaders:
                file_features[loader.name] = loader.load_file(
                    self.output_dir, uid
                )

            file_dur = get_file_duration(self.output_dir, uid)
            if file_dur is None:
                print(f"  WARN: no metadata for {uid}, skipping")
                self.skipped += 1
                continue

            # VTC + VAD segments are inputs to the cut-point algorithm
            vtc_segs = file_features.get("vtc", [])
            vad_segs = file_features.get("vad", [])

            clips, tier_counts = build_clips(
                vtc_segments=vtc_segs,
                vad_segments=vad_segs,
                file_duration=file_dur,
                max_clip_s=self.max_clip_s,
                split_search_s=self.split_search_s,
            )
            for t, n in tier_counts.items():
                self.tier_counts[t] += n

            # Attach per-clip features from each loader
            for idx, clip in enumerate(clips):
                for loader in self._loaders:
                    loader.attach_to_clip(clip, file_features[loader.name])
                self.all_clips.append((uid, audio_path, idx, clip))

        elapsed = time.time() - t0
        n_built = n_files - self.skipped
        print(
            f"\nClip building: {len(self.all_clips)} clips from "
            f"{n_built} files in {elapsed:.1f}s"
        )
        if self.skipped:
            print(f"  skipped: {self.skipped} files")

        return self.all_clips

    def write_shards(self, shard_dir: Path | None = None) -> list[Path]:
        """Write clips to WebDataset tar shards.

        Parameters
        ----------
        shard_dir : Path | None
            Output directory; defaults to ``output_dir / "shards"``.

        Returns
        -------
        List of shard file paths.
        """
        if not self.all_clips:
            raise RuntimeError("No clips to write. Call build_clips() first.")
        if shard_dir is None:
            shard_dir = self.output_dir / "shards"

        print("Writing shards...", flush=True)
        t0 = time.time()
        shard_paths = write_shards(
            clips=self.all_clips,
            output_dir=shard_dir,
            prefix="shards",
            max_shard_clips=self.shard_size,
            audio_fmt=self.audio_fmt,
            target_sr=self.target_sr,
        )
        elapsed = time.time() - t0

        print(f"  Wrote {len(shard_paths)} shards in {elapsed:.1f}s")
        for p in shard_paths:
            size_mb = p.stat().st_size / 1e6
            print(f"    {p.name}  ({size_mb:.1f} MB)")
        return shard_paths

    def write_manifest(self, shard_dir: Path | None = None) -> Path:
        """Write a CSV manifest of all clips (flat, CSV-friendly columns).

        Returns the manifest path.
        """
        from src.core import VTC_LABELS

        if not self.all_clips:
            raise RuntimeError("No clips. Call build_clips() first.")
        if shard_dir is None:
            shard_dir = self.output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        for uid, audio_path, clip_idx, clip in self.all_clips:
            meta = clip.to_metadata(uid, clip_idx)
            # Drop large/nested fields from CSV (kept in per-shard JSON)
            meta.pop("vad_segments", None)
            meta.pop("vtc_segments", None)
            meta.pop("snr", None)
            meta.pop("c50", None)
            meta.pop("esc_profile", None)
            # Flatten nested types for CSV compatibility
            meta["labels_present"] = ";".join(meta.get("labels_present", []))
            ld = meta.pop("label_durations", {})
            for lbl in VTC_LABELS:
                meta[f"dur_{lbl}"] = round(ld.get(lbl, 0.0), 3)
            vc = meta.pop("vad_coverage_by_label", {})
            for lbl in VTC_LABELS:
                meta[f"vad_cov_{lbl}"] = round(vc.get(lbl, 0.0), 3)
            manifest_rows.append(meta)

        clip_manifest = pl.DataFrame(manifest_rows)
        manifest_path = shard_dir / "manifest.csv"
        clip_manifest.write_csv(manifest_path)
        print(f"  Manifest: {manifest_path}  ({len(clip_manifest)} clips)")
        return manifest_path
