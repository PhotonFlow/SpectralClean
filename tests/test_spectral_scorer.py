"""Tests for spectralclean.spectral_scorer."""

import numpy as np
import pytest

from spectralclean.spectral_scorer import SpectralScorer, SpectralResult


class TestSpectralScorer:

    @pytest.fixture
    def embeddings(self):
        """Synthetic embeddings: 3 tight clusters + 2 outliers."""
        rng = np.random.default_rng(42)

        # 3 clusters of 20 samples each
        cluster1 = rng.normal(loc=[1, 0, 0, 0], scale=0.1, size=(20, 4))
        cluster2 = rng.normal(loc=[0, 1, 0, 0], scale=0.1, size=(20, 4))
        cluster3 = rng.normal(loc=[0, 0, 1, 0], scale=0.1, size=(20, 4))

        # 2 outliers
        outliers = rng.normal(loc=[5, 5, 5, 5], scale=0.1, size=(2, 4))

        embeddings = np.vstack([cluster1, cluster2, cluster3, outliers])

        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def test_score_returns_correct_shapes(self, embeddings):
        scorer = SpectralScorer(top_k=2)
        result = scorer.score(embeddings)

        assert isinstance(result, SpectralResult)
        assert result.scores.shape == (62,)
        assert result.eigenvectors.shape == (4, 2)
        assert result.eigenvalues.shape == (2,)
        assert result.projections.shape == (62, 2)

    def test_scores_are_non_negative(self, embeddings):
        scorer = SpectralScorer(top_k=3)
        result = scorer.score(embeddings)
        assert np.all(result.scores >= 0)

    def test_eigenvalues_descending(self, embeddings):
        scorer = SpectralScorer(top_k=4)
        result = scorer.score(embeddings)
        assert np.all(np.diff(result.eigenvalues) <= 0)

    def test_top_k_clipped_to_dimension(self):
        """top_k larger than D should not crash."""
        embeddings = np.random.randn(10, 3)
        scorer = SpectralScorer(top_k=100)
        result = scorer.score(embeddings)
        assert result.eigenvectors.shape[1] <= 3

    def test_prune_fixed_amount(self, embeddings):
        scorer = SpectralScorer(top_k=2)
        clean, low, high = scorer.prune_fixed_amount(embeddings, n_remove=10)

        assert len(clean) + len(low) + len(high) == 62
        assert len(low) + len(high) == 10

    def test_prune_preserves_indices(self, embeddings):
        scorer = SpectralScorer(top_k=2)
        clean, low, high = scorer.prune_fixed_amount(embeddings, n_remove=5)

        all_indices = set(clean.tolist()) | set(low.tolist()) | set(high.tolist())
        assert all_indices == set(range(62))

    def test_rank_by_typicality(self, embeddings):
        scorer = SpectralScorer(top_k=2)
        result = scorer.score(embeddings)
        ranked = scorer.rank_by_typicality(result.scores)

        assert len(ranked) == 62
        # First index should be closest to median
        median = np.median(result.scores)
        assert abs(result.scores[ranked[0]] - median) <= abs(
            result.scores[ranked[-1]] - median
        )

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError):
            SpectralScorer(top_k=0)
