"""Shared fixtures for the VTC test suite.

Session-scoped fixtures create stitched audio files that combine real
audiobook chunks with silence to simulate long-form recordings with
realistic (low) speech ratios.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOOD_BOOK = PROJECT_ROOT / "data" / "vtc_samples" / "good_book"
SHORT_FAIL = PROJECT_ROOT / "data" / "vtc_samples" / "short_fail"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
AUDIO_DIR = FIXTURE_DIR / "audio"
MANIFEST_PATH = FIXTURE_DIR / "test_manifest.csv"

SR = 16_000


# ---------------------------------------------------------------------------
# TenVAD availability check
# ---------------------------------------------------------------------------


def _tenvad_available() -> bool:
    """Return True if TenVAD can initialise on this machine."""
    try:
        from ten_vad import TenVad

        v = TenVad(hop_size=256, threshold=0.5)
        del v
        return True
    except Exception:
        return False


_TENVAD_OK = _tenvad_available()

requires_tenvad = pytest.mark.skipif(
    not _TENVAD_OK,
    reason="TenVAD unavailable (needs 'module load llvm' for libc++.so.1) — run via scripts/test.slurm",
)


# ---------------------------------------------------------------------------
# Audio stitching helper
# ---------------------------------------------------------------------------


def _stitch_with_silence(
    wav_paths: list[Path],
    silence_between_s: float,
    output_path: Path,
) -> tuple[float, list[tuple[float, float]]]:
    """Concatenate WAV files with silence gaps.

    Returns ``(total_duration_s, speech_regions)`` where speech_regions is
    a list of ``(onset, offset)`` pairs marking where original audio was
    placed.
    """
    silence_samples = int(silence_between_s * SR)
    silence = np.zeros(silence_samples, dtype=np.int16)

    all_samples: list[np.ndarray] = []
    speech_regions: list[tuple[float, float]] = []
    cursor = 0  # sample index

    for i, wav in enumerate(wav_paths):
        # Leading silence (except before the first file)
        if i > 0:
            all_samples.append(silence)
            cursor += silence_samples

        data, file_sr = sf.read(wav, dtype="int16")
        if data.ndim == 2:
            data = data[:, 0]

        if file_sr != SR:
            # Simple linear resample
            n_out = int(len(data) * SR / file_sr)
            indices = np.linspace(0, len(data) - 1, n_out)
            lo = np.floor(indices).astype(np.int64)
            hi = np.minimum(lo + 1, len(data) - 1)
            frac = (indices - lo).astype(np.float32)
            data = (data[lo] * (1 - frac) + data[hi] * frac).astype(np.int16)

        onset_s = round(cursor / SR, 3)
        all_samples.append(data)
        cursor += len(data)
        offset_s = round(cursor / SR, 3)
        speech_regions.append((onset_s, offset_s))

    # Trailing silence (make the file ~20% longer)
    trailing = int(cursor * 0.2)
    all_samples.append(np.zeros(trailing, dtype=np.int16))
    cursor += trailing

    stitched = np.concatenate(all_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), stitched, SR, subtype="PCM_16")
    return round(cursor / SR, 3), speech_regions


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def stitched_audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create stitched long-form audio files and a test manifest.

    Creates three test files:
    - ``long_low_speech.wav``:  6 good_book chunks with 60s silence gaps
      (speech ratio ~30 %)
    - ``long_high_speech.wav``: 6 good_book chunks with 5s silence gaps
      (speech ratio ~90 %)
    - ``short_file.wav``:       a single short_fail file, no stitching

    Returns the fixture audio directory. Also writes
    ``test_manifest.csv`` in the same directory.
    """
    fixture_dir = tmp_path_factory.mktemp("vtc_test_audio")
    audio_dir = fixture_dir / "audio"
    audio_dir.mkdir()

    good_wavs = sorted(GOOD_BOOK.glob("*.wav"))
    short_wavs = sorted(SHORT_FAIL.glob("*.wav"))

    if not good_wavs:
        pytest.skip("No test audio in data/vtc_samples/good_book/")
    if not short_wavs:
        pytest.skip("No test audio in data/vtc_samples/short_fail/")

    rows: list[dict] = []

    # 1. Low speech ratio (~30 %)
    dur, regions = _stitch_with_silence(
        good_wavs[:6],
        silence_between_s=60.0,
        output_path=audio_dir / "long_low_speech.wav",
    )
    rows.append({"path": str(audio_dir / "long_low_speech.wav")})

    # 2. High speech ratio (~90 %)
    dur, regions = _stitch_with_silence(
        good_wavs[:6],
        silence_between_s=5.0,
        output_path=audio_dir / "long_high_speech.wav",
    )
    rows.append({"path": str(audio_dir / "long_high_speech.wav")})

    # 3. Short file (copy as-is)
    short_out = audio_dir / "short_file.wav"
    shutil.copy2(short_wavs[0], short_out)
    rows.append({"path": str(short_out)})

    # Write manifest
    manifest = fixture_dir / "test_manifest.csv"
    pl.DataFrame(rows).write_csv(manifest)

    return fixture_dir


@pytest.fixture(scope="session")
def good_book_wavs() -> list[Path]:
    """Return sorted list of good_book WAV paths."""
    wavs = sorted(GOOD_BOOK.glob("*.wav"))
    if not wavs:
        pytest.skip("No test audio in data/vtc_samples/good_book/")
    return wavs


@pytest.fixture(scope="session")
def short_fail_wavs() -> list[Path]:
    """Return sorted list of short_fail WAV paths."""
    wavs = sorted(SHORT_FAIL.glob("*.wav"))
    if not wavs:
        pytest.skip("No test audio in data/vtc_samples/short_fail/")
    return wavs
