"""Tests for the PANNs ESC pipeline.

Tests are split into two groups:

1. **Pure-numpy tests** for ``pool_esc``, ``pool_to_categories``,
   ``map_to_categories`` — run anywhere (login node included).
2. **GPU/model tests** for ``extract_panns`` and full inference — require
   PANNs + GPU (run via ``sbatch slurm/test.slurm``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.esc import (
    CATEGORIES,
    _SPEECH_INDICES,
    map_to_categories,
    pool_esc,
    pool_to_categories,
)


# ======================================================================
# map_to_categories — pure numpy, no GPU needed
# ======================================================================


class TestMapToCategories:
    """Unit tests for the AudioSet→coarse category mapping."""

    def test_returns_all_categories(self):
        """All defined categories must be present in the output."""
        probs = np.random.rand(527).astype(np.float32)
        result = map_to_categories(probs)

        for cat in CATEGORIES:
            assert cat in result, f"Missing category: {cat}"

    def test_values_in_range(self):
        """Each category probability should be in [0, 1]."""
        probs = np.random.rand(527).astype(np.float32)
        result = map_to_categories(probs)

        for cat, val in result.items():
            assert 0.0 <= val <= 1.0, f"{cat}: {val} out of range"

    def test_zero_probs_yield_zero_categories(self):
        """All-zero input → all categories are zero."""
        probs = np.zeros(527, dtype=np.float32)
        result = map_to_categories(probs)

        for cat, val in result.items():
            assert val == 0.0, f"{cat} should be 0.0 with zero input"

    def test_speech_indices_excluded(self):
        """Speech-related indices should not leak into ESC categories."""
        # Set only speech indices to 1.0, rest to 0.0
        probs = np.zeros(527, dtype=np.float32)
        for idx in _SPEECH_INDICES:
            probs[idx] = 1.0
        result = map_to_categories(probs)

        # All categories should be 0 since only speech indices are active
        for cat, val in result.items():
            assert val == 0.0, f"{cat}={val} — speech indices leaked through"


# ======================================================================
# pool_esc — pure numpy
# ======================================================================


class TestPoolNoise:
    """Unit tests for the ESC pooling helper."""

    def test_passthrough_when_pool_gte_step(self):
        """When pool_window >= step, just convert dtype."""
        raw = np.random.rand(5, 527).astype(np.float32)
        pooled, step = pool_esc(raw, step_s=10.0, pool_window_s=10.0)

        assert pooled.dtype == np.float16
        assert pooled.shape == (5, 527)
        assert step == 10.0

    def test_upsample_when_pool_smaller(self):
        """When pool_window < step, repeat to approximate finer bins."""
        raw = np.ones((3, 527), dtype=np.float32) * 0.5
        pooled, step = pool_esc(raw, step_s=10.0, pool_window_s=1.0)

        assert pooled.dtype == np.float16
        # 3 windows × 10x repeat = 30 bins
        assert pooled.shape[0] == 30
        assert pooled.shape[1] == 527
        assert step == 1.0

    def test_output_values_preserved(self):
        """Values should be preserved through pooling."""
        raw = np.ones((2, 527), dtype=np.float32) * 0.7
        pooled, _ = pool_esc(raw, step_s=10.0, pool_window_s=10.0)

        np.testing.assert_allclose(pooled.astype(np.float32), 0.7, atol=0.01)


# ======================================================================
# pool_to_categories — pure numpy
# ======================================================================


class TestPoolToCategories:
    """Test the category-level pooling function."""

    def test_output_shape(self):
        """Output should be (n_bins, n_categories)."""
        raw = np.random.rand(5, 527).astype(np.float32)
        pooled, cats, step = pool_to_categories(raw, step_s=10.0, pool_window_s=10.0)

        assert pooled.shape == (5, len(CATEGORIES))
        assert cats == CATEGORIES
        assert step == 10.0

    def test_dtype_is_float16(self):
        """Output dtype should be float16 for compact storage."""
        raw = np.random.rand(3, 527).astype(np.float32)
        pooled, _, _ = pool_to_categories(raw, step_s=10.0, pool_window_s=10.0)

        assert pooled.dtype == np.float16

    def test_categories_match(self):
        """Categories should match the module-level CATEGORIES list."""
        raw = np.random.rand(2, 527).astype(np.float32)
        _, cats, _ = pool_to_categories(raw, step_s=10.0, pool_window_s=10.0)

        assert cats == CATEGORIES

    def test_upsample_multiplies_rows(self):
        """Pool to finer resolution multiplies rows."""
        raw = np.ones((2, 527), dtype=np.float32) * 0.3
        pooled, _, step = pool_to_categories(raw, step_s=10.0, pool_window_s=1.0)

        assert pooled.shape[0] == 20  # 2 × 10
        assert step == 1.0


# ======================================================================
# Clip integration — esc_array on Clip dataclass
# ======================================================================


class TestClipNoiseIntegration:
    """Test ESC fields on the Clip dataclass."""

    def test_esc_profile_none_when_no_data(self):
        """esc_profile should be None when no ESC data attached."""
        from src.packaging.clips import Clip

        clip = Clip(abs_onset=0.0, abs_offset=10.0)
        assert clip.esc_profile is None
        assert clip.dominant_esc is None

    def test_esc_profile_with_data(self):
        """esc_profile should return mean per category."""
        from src.packaging.clips import Clip

        cats = CATEGORIES
        n = len(cats)
        esc_arr = np.zeros((10, n), dtype=np.float16)
        # Make "music" (cats.index("music")) highest
        music_idx = cats.index("music")
        esc_arr[:, music_idx] = 0.9

        clip = Clip(abs_onset=0.0, abs_offset=10.0)
        clip.esc_array = esc_arr
        clip.esc_categories = cats

        profile = clip.esc_profile
        assert profile is not None
        assert profile["music"] == pytest.approx(0.9, abs=0.05)
        assert clip.dominant_esc == "music"


# ======================================================================
# GPU / model tests — need compute node
# ======================================================================


def _panns_available() -> bool:
    """Return True if PANNs can run inference."""
    try:
        import torch
        from panns_inference import AudioTagging
        # Just check the import works — don't actually load the model
        return True
    except Exception:
        return False


requires_panns = pytest.mark.skipif(
    not _panns_available(),
    reason="PANNs/torch unavailable — run via slurm/test.slurm on a GPU node",
)


@requires_panns
class TestExtractPanns:
    """Integration tests: run PANNs on real audio fixtures."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        """Load PANNs model once per test class."""
        import torch
        from panns_inference import AudioTagging

        from src.utils import set_seeds
        set_seeds(42)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.at = AudioTagging(checkpoint_path=None, device=dev)

    def test_returns_correct_shape(self, speech_clean_wav: Path):
        """extract_panns returns (n_windows, 527) float32 array."""
        from src.pipeline.esc import extract_panns

        probs, step = extract_panns(self.at, speech_clean_wav, window_s=10.0)

        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float32
        assert probs.ndim == 2
        assert probs.shape[1] == 527
        assert probs.shape[0] >= 1  # at least 1 window
        assert step == 10.0

    def test_probabilities_in_range(self, speech_clean_wav: Path):
        """All probabilities should be in [0, 1]."""
        from src.pipeline.esc import extract_panns

        probs, _ = extract_panns(self.at, speech_clean_wav, window_s=10.0)

        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_speech_detected_in_speech_file(self, speech_clean_wav: Path):
        """Speech class (index 0) should have high probability for speech audio."""
        from src.pipeline.esc import extract_panns

        probs, _ = extract_panns(self.at, speech_clean_wav, window_s=10.0)

        # Index 0 is "Speech" in AudioSet
        speech_prob = float(probs[:, 0].mean())
        assert speech_prob > 0.3, f"Speech prob={speech_prob} too low for speech file"

    def test_silence_file(self, fixture_dir: Path):
        """Silence file should have low probabilities across categories."""
        from src.pipeline.esc import extract_panns

        silence_wav = fixture_dir / "silence.wav"
        if not silence_wav.exists():
            pytest.skip("silence.wav not found")

        probs, _ = extract_panns(self.at, silence_wav, window_s=10.0)

        # Mean across all non-speech categories should be low
        non_speech_mean = float(np.mean(probs[:, 16:]))
        assert non_speech_mean < 0.3, f"Non-speech mean={non_speech_mean} too high for silence"

    def test_deterministic(self, speech_clean_wav: Path):
        """Two runs with same seed should give identical results."""
        from src.pipeline.esc import extract_panns
        from src.utils import set_seeds

        set_seeds(42)
        p1, _ = extract_panns(self.at, speech_clean_wav, window_s=10.0)

        set_seeds(42)
        p2, _ = extract_panns(self.at, speech_clean_wav, window_s=10.0)

        np.testing.assert_array_equal(p1, p2)

    def test_full_pipeline_npz_format(self, speech_clean_wav: Path, tmp_path: Path):
        """End-to-end: extract → pool → save → reload → verify."""
        from src.pipeline.esc import extract_panns, pool_to_categories

        probs, step = extract_panns(self.at, speech_clean_wav, window_s=10.0)
        pooled, cats, pool_step = pool_to_categories(probs, step, pool_window_s=1.0)

        # Save
        out = tmp_path / "test.npz"
        np.savez_compressed(
            out,
            categories=pooled,
            category_names=np.array(cats),
            pool_step_s=np.float32(pool_step),
            inference_step_s=np.float32(step),
            n_inference_windows=np.int32(probs.shape[0]),
        )

        # Reload
        data = np.load(out, allow_pickle=True)
        loaded = data["categories"].astype(np.float32)
        loaded_cats = list(data["category_names"])

        assert loaded.shape[1] == len(CATEGORIES)
        assert loaded_cats == CATEGORIES
        np.testing.assert_allclose(loaded, pooled.astype(np.float32), atol=0.01)
