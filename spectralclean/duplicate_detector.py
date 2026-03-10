"""
Semantic duplicate detection via embedding distance.

Identifies near-duplicate object instances within a dataset by computing
pairwise distances in CLIP embedding space and flagging pairs below a
configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


@dataclass
class DuplicateReport:
    """Result of duplicate detection."""

    duplicate_indices: Set[int]
    """Indices (into the original path list) of duplicates to remove."""

    distance_matrix: np.ndarray
    """Full pairwise distance matrix (upper-triangle only is meaningful)."""

    n_total: int
    """Total number of images analysed."""

    n_duplicates: int
    """Number of duplicates found."""


class DuplicateDetector:
    """Find semantic duplicates in a set of embeddings.

    Parameters
    ----------
    threshold : float
        Euclidean distance below which two samples are considered
        duplicates.  Suitable range for L2-normalised CLIP embeddings
        is typically 0.05–0.25.
    """

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold

    def detect(
        self,
        embeddings: np.ndarray,
    ) -> DuplicateReport:
        """Identify duplicate indices.

        For each group of near-duplicates, keeps the *first* sample
        (by index) and marks the rest for removal.

        Parameters
        ----------
        embeddings : ndarray (N, D)
            L2-normalised feature vectors.

        Returns
        -------
        DuplicateReport
        """
        dists = euclidean_distances(embeddings)
        np.fill_diagonal(dists, np.inf)

        marked: Set[int] = set()

        for i in range(len(embeddings)):
            if i in marked:
                continue
            neighbours = np.where(dists[i] < self.threshold)[0]
            for n in neighbours:
                if n not in marked and n != i:
                    marked.add(n)

        return DuplicateReport(
            duplicate_indices=marked,
            distance_matrix=dists,
            n_total=len(embeddings),
            n_duplicates=len(marked),
        )
