"""Tests for stitched audio test fixtures and end-to-end VAD on long-form audio."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import soundfile as sf

pytest.importorskip(
    "torchcodec",
    reason="torchcodec/ffmpeg not available — run via scripts/test.slurm on a compute node",
)

from scripts.core.regions import (
    activity_region_coverage,
    merge_into_activity_regions,
)
from tests.conftest import requires_tenvad

from scripts.core.vad_processing import process_file


class TestStitchedFixtures:
    """Verify the stitched audio files exist and have expected properties."""

    def test_files_created(self, stitched_audio_dir: Path):
        audio_dir = stitched_audio_dir / "audio"
        assert (audio_dir / "long_low_speech.wav").exists()
        assert (audio_dir / "long_high_speech.wav").exists()
        assert (audio_dir / "short_file.wav").exists()

    def test_manifest_created(self, stitched_audio_dir: Path):
        manifest = stitched_audio_dir / "test_manifest.csv"
        assert manifest.exists()
        df = pl.read_csv(manifest)
        assert len(df) == 3
        assert "path" in df.columns

    def test_low_speech_longer(self, stitched_audio_dir: Path):
        """Low-speech file (60s gaps) should be longer than high-speech (5s gaps)."""
        low = sf.info(str(stitched_audio_dir / "audio" / "long_low_speech.wav"))
        high = sf.info(str(stitched_audio_dir / "audio" / "long_high_speech.wav"))
        assert low.duration > high.duration

    def test_audio_properties(self, stitched_audio_dir: Path):
        for name in ["long_low_speech.wav", "long_high_speech.wav", "short_file.wav"]:
            info = sf.info(str(stitched_audio_dir / "audio" / name))
            assert info.samplerate == 16_000
            assert info.channels == 1


@requires_tenvad
class TestVadOnStitchedAudio:
    """End-to-end: VAD on stitched files should detect speech in the right regions."""

    def test_low_speech_ratio(self, stitched_audio_dir: Path):
        """Low-speech file should have a speech ratio below 50%."""
        wav = stitched_audio_dir / "audio" / "long_low_speech.wav"
        meta, segs = process_file((wav, 256, 0.5))
        assert meta["success"] is True
        assert meta["speech_ratio"] < 0.5
        assert meta["n_speech_segments"] > 0

    def test_high_speech_ratio(self, stitched_audio_dir: Path):
        """Dense file (5s gaps) should have a higher speech ratio than sparse file (60s gaps)."""
        low_wav = stitched_audio_dir / "audio" / "long_low_speech.wav"
        high_wav = stitched_audio_dir / "audio" / "long_high_speech.wav"
        low_meta, _ = process_file((low_wav, 256, 0.5))
        high_meta, _ = process_file((high_wav, 256, 0.5))
        assert high_meta["success"] is True
        assert high_meta["speech_ratio"] > low_meta["speech_ratio"], (
            f"Dense file ratio {high_meta['speech_ratio']:.3f} should exceed "
            f"sparse file ratio {low_meta['speech_ratio']:.3f}"
        )

    def test_short_file_no_error(self, stitched_audio_dir: Path):
        wav = stitched_audio_dir / "audio" / "short_file.wav"
        meta, segs = process_file((wav, 256, 0.5))
        assert meta["success"] is True


@requires_tenvad
class TestActivityRegionsOnStitched:
    """Test that activity regions correctly identify sparse speech in stitched audio."""

    def test_low_speech_has_multiple_regions(self, stitched_audio_dir: Path):
        """Low-speech file: VAD segments → activity regions should cover < 90 %."""
        wav = stitched_audio_dir / "audio" / "long_low_speech.wav"
        meta, segs = process_file((wav, 256, 0.5))
        assert meta["success"] is True

        file_dur = meta["duration"]
        vad_pairs = [(s["onset"], s["offset"]) for s in segs]
        regions = merge_into_activity_regions(vad_pairs, file_dur)
        coverage = activity_region_coverage(regions, file_dur)

        # With 60s silence gaps, coverage should be well under 90%
        assert coverage < 0.9, (
            f"Expected coverage < 0.9, got {coverage:.2f} "
            f"({len(regions)} regions over {file_dur:.0f}s)"
        )
        assert len(regions) >= 2  # Multiple speech clusters

    def test_high_speech_coverage_above_threshold(self, stitched_audio_dir: Path):
        """Dense file (5s gaps) should have higher activity region coverage than sparse file (60s gaps)."""
        low_wav = stitched_audio_dir / "audio" / "long_low_speech.wav"
        high_wav = stitched_audio_dir / "audio" / "long_high_speech.wav"

        def get_coverage(wav: Path) -> float:
            meta, segs = process_file((wav, 256, 0.5))
            assert meta["success"] is True
            vad_pairs = [(s["onset"], s["offset"]) for s in segs]
            regions = merge_into_activity_regions(vad_pairs, meta["duration"])
            return activity_region_coverage(regions, meta["duration"])

        low_cov = get_coverage(low_wav)
        high_cov = get_coverage(high_wav)
        assert high_cov > low_cov, (
            f"Dense file coverage {high_cov:.2f} should exceed "
            f"sparse file coverage {low_cov:.2f}"
        )
