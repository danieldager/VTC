"""End-to-end tests for the two-step DL++ workflow:

  1. ``prepare()``  — feature extraction + packaging → WebDataset shards
  2. ``create_dataloader()`` — config + shards → PyTorch DataLoader

These tests exercise the full pipeline against the seedlings_1 dataset.
They require shards to already exist (produced by the pipeline or a prior
``prepare()`` call).  They are deliberately read-only: they never submit new
SLURM jobs or modify any pipeline outputs.

Run via SLURM (needed for webdataset + heavy deps)::

    sbatch slurm/test.slurm tests/test_create_dataloader.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dataloader.batch.data_batch import DataBatch
from dataloader.config import DatasetConfig, FilterConfig, LoaderConfig, PipelineConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDLINGS_DIR = PROJECT_ROOT / "output" / "seedlings_1"
SHARD_DIR = SEEDLINGS_DIR / "shards"

# Skip the entire module if seedlings_1 shards are missing.
pytestmark = pytest.mark.skipif(
    not SHARD_DIR.exists() or not list(SHARD_DIR.glob("*.tar")),
    reason=(
        "output/seedlings_1/shards/ not found — run the pipeline first:\n"
        "  python -m dataloader.prepare <config.json>\n"
        "  OR: bash slurm/pipeline.sh seedlings_1"
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(**overrides) -> DatasetConfig:
    """Build a lightweight test config (no GPU, no multiprocessing)."""
    defaults = dict(
        dataset_dir=str(SEEDLINGS_DIR),
        pipeline=PipelineConfig(target_sr=16000),
        loader=LoaderConfig(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        ),
    )
    defaults.update(overrides)
    return DatasetConfig(**defaults)  # type: ignore


# ── prepare() idempotency (no SLURM submission) ───────────────────────────────


class TestPrepareIdempotency:
    """Verify prepare() short-circuits when outputs already exist."""

    def test_skips_when_shards_exist(self, capsys) -> None:
        """prepare() should print a 'nothing to do' message and return fast."""
        from dataloader.prepare import prepare

        prepare(_make_config())  # shards exist → must not submit anything
        captured = capsys.readouterr()
        assert "already have" in captured.out.lower() or "nothing to do" in captured.out.lower()

    def test_stage_detection_vad(self) -> None:
        """_stage_done() correctly detects VAD completion."""
        from dataloader.prepare import _stage_done
        assert _stage_done(SEEDLINGS_DIR, "vad"), \
            "VAD outputs not found — pipeline may not have run"

    def test_stage_detection_vtc(self) -> None:
        from dataloader.prepare import _stage_done
        assert _stage_done(SEEDLINGS_DIR, "vtc"), \
            "VTC outputs not found — pipeline may not have run"

    def test_stage_detection_snr(self) -> None:
        from dataloader.prepare import _stage_done
        assert _stage_done(SEEDLINGS_DIR, "snr"), \
            "SNR outputs not found — pipeline may not have run"

    def test_stage_detection_esc(self) -> None:
        from dataloader.prepare import _stage_done
        assert _stage_done(SEEDLINGS_DIR, "esc"), \
            "Noise outputs not found — pipeline may not have run"

    def test_stage_detection_package(self) -> None:
        from dataloader.prepare import _stage_done
        assert _stage_done(SEEDLINGS_DIR, "package"), \
            "Package outputs not found — shards missing"


# ── DatasetConfig serialization ───────────────────────────────────────────────


class TestDatasetConfig:
    """Verify DatasetConfig save / load round-trip."""

    def test_round_trip(self, tmp_path: Path) -> None:
        config = _make_config(
            filters=FilterConfig(min_snr_db=5.0, required_labels=["KCHI"]),
        )
        path = tmp_path / "config.json"
        config.save(path)
        loaded = DatasetConfig.load(path)

        assert loaded.dataset_dir == config.dataset_dir
        assert loaded.filters.min_snr_db == 5.0
        assert loaded.filters.required_labels == ["KCHI"]
        assert loaded.loader.batch_size == 2
        assert loaded.loader.audio_key == "wav"
        assert loaded.pipeline.target_sr == 16000

    def test_round_trip_json_structure(self, tmp_path: Path) -> None:
        config = _make_config()
        path = tmp_path / "config.json"
        config.save(path)
        data = json.loads(path.read_text())
        assert {"dataset_dir", "pipeline", "filters", "loader"} <= set(data)

    def test_load_from_directory(self, tmp_path: Path) -> None:
        config = _make_config()
        config.save(tmp_path)  # saves as dataset_config.json inside dir
        loaded = DatasetConfig.load(tmp_path)
        assert loaded.dataset_dir == config.dataset_dir

    def test_filters_is_active(self) -> None:
        assert not FilterConfig().is_active
        assert FilterConfig(min_snr_db=5.0).is_active
        assert FilterConfig(required_labels=["KCHI"]).is_active


# ── create_dataloader() end-to-end streaming ─────────────────────────────────


class TestCreateDataloader:
    """End-to-end: config → DataLoader → DataBatch."""

    def test_unfiltered_yields_valid_batch(self) -> None:
        """Basic check: unfiltered loader streams valid DataBatch objects."""
        from dataloader.create import create_dataloader

        loader = create_dataloader(_make_config())
        batch = next(iter(loader))

        assert isinstance(batch, DataBatch)
        assert batch.waveforms.ndim == 3                       # (B, C, T)
        assert batch.waveforms.shape[0] <= 2                   # batch size
        assert batch.waveforms.shape[1] == 1                   # mono
        assert batch.sample_rate == 16000
        assert batch.attention_mask.shape[0] == batch.waveforms.shape[0]
        assert len(batch.wav_ids) == batch.batch_size
        assert all(isinstance(wid, str) and len(wid) > 0 for wid in batch.wav_ids)

    def test_batch_metadata_present(self) -> None:
        """Metadata from the .json sidecar flows through to DataBatch."""
        from dataloader.create import create_dataloader

        loader = create_dataloader(_make_config())
        batch = next(iter(loader))

        assert batch.durations_s is not None
        assert batch.durations_s.shape == (batch.batch_size,)
        assert (batch.durations_s > 0).all()

        for meta in batch.metadata:
            assert isinstance(meta, dict)
            assert "wav_id" in meta

    def test_waveform_lengths_match_mask(self) -> None:
        """Attention mask True-count equals waveform_lengths."""
        from dataloader.create import create_dataloader

        loader = create_dataloader(_make_config())
        batch = next(iter(loader))

        for i in range(batch.batch_size):
            mask_len = batch.attention_mask[i].sum().item()
            declared = batch.waveform_lengths[i].item()
            assert mask_len == declared, (
                f"Sample {i}: mask says {mask_len} but waveform_lengths says {declared}"
            )

    def test_filtered_restricts_sources(self) -> None:
        """Filtered loader only yields clips from allowed source files."""
        from dataloader.build import build_manifest
        from dataloader.create import create_dataloader

        filters = FilterConfig(min_snr_db=5.0)
        manifest = build_manifest(SEEDLINGS_DIR, filters=filters)
        allowed = set(manifest.df["wav_id"].to_list())
        assert len(allowed) > 0, "Filter too aggressive — no files pass"

        loader = create_dataloader(_make_config(filters=filters))
        uids_seen: set[str] = set()
        for i, batch in enumerate(loader):
            for meta in batch.metadata:
                uid = meta.get("uid", "")
                if uid:
                    uids_seen.add(uid)
            if i >= 4:
                break

        for uid in uids_seen:
            assert uid in allowed, (
                f"UID {uid!r} was streamed but is not in the allowed set"
            )

    def test_batch_to_device(self) -> None:
        """DataBatch.to('cpu') completes without error."""
        from dataloader.create import create_dataloader

        loader = create_dataloader(_make_config())
        batch = next(iter(loader)).to("cpu")
        assert batch.waveforms.device == torch.device("cpu")

    def test_create_from_saved_config_path(self, tmp_path: Path) -> None:
        """create_dataloader() accepts a filesystem path to a saved config."""
        from dataloader.create import create_dataloader

        config = _make_config()
        path = tmp_path / "config.json"
        config.save(path)

        loader = create_dataloader(path)
        batch = next(iter(loader))
        assert isinstance(batch, DataBatch)
        assert batch.batch_size > 0

    def test_no_shards_raises(self, tmp_path: Path) -> None:
        """create_dataloader() raises FileNotFoundError for empty shard dir."""
        from dataloader.create import create_dataloader

        config = _make_config(dataset_dir=str(tmp_path))
        (tmp_path / "shards").mkdir()   # empty shard dir
        with pytest.raises(FileNotFoundError, match="No .tar shards"):
            create_dataloader(config)


