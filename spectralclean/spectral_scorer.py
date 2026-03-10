"""
Spectral scoring via eigenvector projection.

Projects embedding vectors onto the top-K eigenvectors of their Gram matrix
to produce per-sample "spectral scores".  Samples with extreme scores
(very low or very high) are likely noise or redundant duplicates.

Reference
---------
Zhu, Z. et al. "Detecting Corrupted Labels Without Training a Model to
Classify" (FINE, 2022).  This module extends the single-eigenvector approach
to a multi-eigenvector (top-K) subspace for improved separation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SpectralResult:
    """Container for the outputs of spectral scoring."""

    scores: np.ndarray
    """Per-sample spectral scores — shape ``(N,)``."""

    eigenvectors: np.ndarray
    """Top-K eigenvectors of the Gram matrix — shape ``(D, K)``."""

    eigenvalues: np.ndarray
    """Corresponding eigenvalues — shape ``(K,)``."""

    projections: np.ndarray
    """Per-sample projections onto the top-K subspace — shape ``(N, K)``."""


class SpectralScorer:
    """Compute spectral scores from embedding vectors.

    Parameters
    ----------
    top_k : int
        Number of leading eigenvectors to use.  Higher values capture more
        variance but may dilute the noise signal.  Typical values: 1–8.
    """

    def __init__(self, top_k: int = 4) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        self._eigenvectors: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, embeddings: np.ndarray) -> SpectralResult:
        """Compute spectral scores for every row in *embeddings*.

        Parameters
        ----------
        embeddings : ndarray of shape ``(N, D)``
            L2-normalised feature vectors.

        Returns
        -------
        SpectralResult
            Contains ``scores``, ``eigenvectors``, ``eigenvalues``,
            and ``projections``.
        """
        eigenvectors, eigenvalues = self._compute_eigenvectors(embeddings)

        k = min(self.top_k, eigenvectors.shape[1])
        top_vectors = eigenvectors[:, :k]

        # Project: (N, D) @ (D, K) -> (N, K)
        projections = embeddings @ top_vectors

        # Score = squared L2 norm of the projection vector
        scores = np.sum(projections ** 2, axis=1)

        return SpectralResult(
            scores=scores,
            eigenvectors=top_vectors,
            eigenvalues=eigenvalues[:k],
            projections=projections,
        )

    def rank_by_typicality(
        self, scores: np.ndarray
    ) -> np.ndarray:
        """Return indices sorted from *most typical* to *most anomalous*.

        Samples near the median score are considered typical; samples at
        either extreme (low = outlier/noise, high = redundant/dominant)
        are considered anomalous.

        Returns
        -------
        ndarray of int
            Indices into the original array, sorted by distance from
            the score median (ascending — most typical first).
        """
        median = np.median(scores)
        deviation = np.abs(scores - median)
        return np.argsort(deviation)

    # ------------------------------------------------------------------
    # Pruning helpers
    # ------------------------------------------------------------------

    def prune_fixed_amount(
        self,
        embeddings: np.ndarray,
        n_remove: int,
        low_ratio: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove a fixed number of samples from both score extremes.

        Parameters
        ----------
        embeddings : ndarray (N, D)
        n_remove : int
            Total number of samples to prune.
        low_ratio : float
            Fraction of *n_remove* to take from the low-score end
            (outliers/noise).  The remainder is taken from the high-score
            end (redundant/dominant).

        Returns
        -------
        clean_indices, low_indices, high_indices : ndarray
            Integer index arrays into the original embedding matrix.
        """
        result = self.score(embeddings)
        sorted_indices = np.argsort(result.scores)

        n_low = int(n_remove * low_ratio)
        n_high = n_remove - n_low

        low_indices = sorted_indices[:n_low] if n_low > 0 else np.array([], dtype=int)
        high_indices = (
            sorted_indices[-n_high:] if n_high > 0 else np.array([], dtype=int)
        )

        remove_set = set(low_indices.tolist()) | set(high_indices.tolist())
        clean_indices = np.array(
            [i for i in range(len(embeddings)) if i not in remove_set]
        )

        return clean_indices, low_indices, high_indices

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_eigenvectors(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Eigen-decomposition of the Gram matrix ``E^T E``."""
        gram = embeddings.T @ embeddings
        eigenvalues, eigenvectors = np.linalg.eigh(gram)

        # Sort descending by eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self._eigenvectors = eigenvectors
        self._eigenvalues = eigenvalues
        return eigenvectors, eigenvalues
