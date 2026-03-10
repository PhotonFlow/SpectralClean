"""Tests for spectralclean.noise_separator."""

import numpy as np
import pytest

from spectralclean.noise_separator import NoiseSeparator, SeparationResult


class TestNoiseSeparator:

    def test_separates_bimodal_distribution(self):
        """Clear bimodal data should be separated cleanly."""
        rng = np.random.default_rng(42)
        clean = rng.normal(loc=1.0, scale=0.3, size=80)
        noisy = rng.normal(loc=5.0, scale=0.5, size=20)
        scores = np.concatenate([clean, noisy])

        sep = NoiseSeparator(threshold=0.5)
        result = sep.separate(scores)

        assert isinstance(result, SeparationResult)
        # Most of the first 80 should be clean
        assert len(result.clean_indices) > 60
        # Most of the last 20 should be noisy
        assert len(result.noisy_indices) > 10
        # Total should match
        assert len(result.clean_indices) + len(result.noisy_indices) == 100

    def test_clean_mean_less_than_noisy(self):
        rng = np.random.default_rng(42)
        scores = np.concatenate([
            rng.normal(0.5, 0.1, 50),
            rng.normal(3.0, 0.2, 50),
        ])
        result = NoiseSeparator().separate(scores)
        assert result.clean_mean < result.noisy_mean

    def test_probs_shape(self):
        scores = np.random.randn(50)
        result = NoiseSeparator().separate(scores)
        assert result.clean_probs.shape == (50,)

    def test_threshold_controls_strictness(self):
        rng = np.random.default_rng(42)
        scores = np.concatenate([
            rng.normal(1.0, 0.5, 70),
            rng.normal(3.0, 0.5, 30),
        ])

        strict = NoiseSeparator(threshold=0.9).separate(scores)
        lenient = NoiseSeparator(threshold=0.3).separate(scores)

        # Stricter threshold → fewer clean samples
        assert len(strict.clean_indices) <= len(lenient.clean_indices)
