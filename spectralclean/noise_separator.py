"""
GMM-based noise separation.

Uses a two-component Gaussian Mixture Model to separate spectral scores
into "clean" and "noisy" populations.  The component with the *lower*
mean is assumed to represent clean data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class SeparationResult:
    """Outcome of GMM-based noise separation."""

    clean_indices: np.ndarray
    """Indices of samples classified as clean."""

    noisy_indices: np.ndarray
    """Indices of samples classified as noisy."""

    clean_probs: np.ndarray
    """Per-sample probability of belonging to the clean component."""

    clean_mean: float
    """Mean spectral score of the clean component."""

    noisy_mean: float
    """Mean spectral score of the noisy component."""


class NoiseSeparator:
    """Split samples into clean/noisy sets using GMM on spectral scores.

    Parameters
    ----------
    threshold : float
        Minimum probability of belonging to the clean component for a
        sample to be kept.  Lower values are more permissive (keep more
        borderline samples).
    n_components : int
        Number of GMM components (default 2: clean vs noisy).
    """

    def __init__(
        self,
        threshold: float = 0.45,
        n_components: int = 2,
    ) -> None:
        self.threshold = threshold
        self.n_components = n_components

    def separate(self, scores: np.ndarray) -> SeparationResult:
        """Separate *scores* into clean and noisy index sets.

        Parameters
        ----------
        scores : ndarray of shape ``(N,)``
            Spectral scores (e.g. from :class:`SpectralScorer`).

        Returns
        -------
        SeparationResult
        """
        X = scores.reshape(-1, 1)

        gmm = GaussianMixture(
            n_components=self.n_components, random_state=42
        )
        gmm.fit(X)

        means = gmm.means_.flatten()
        clean_comp = int(np.argmin(means))
        noisy_comp = int(np.argmax(means))

        probs = gmm.predict_proba(X)
        clean_probs = probs[:, clean_comp]

        is_clean = clean_probs > self.threshold

        clean_indices = np.where(is_clean)[0]
        noisy_indices = np.where(~is_clean)[0]

        return SeparationResult(
            clean_indices=clean_indices,
            noisy_indices=noisy_indices,
            clean_probs=clean_probs,
            clean_mean=float(means[clean_comp]),
            noisy_mean=float(means[noisy_comp]),
        )
