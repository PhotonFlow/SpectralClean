"""Tests for spectralclean.duplicate_detector."""

import numpy as np
import pytest

from spectralclean.duplicate_detector import DuplicateDetector, DuplicateReport


class TestDuplicateDetector:

    def test_exact_duplicates(self):
        """Identical embeddings should be flagged."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # exact dup of 0
            [0.0, 1.0, 0.0],
        ])
        det = DuplicateDetector(threshold=0.01)
        report = det.detect(embeddings)

        assert report.n_duplicates == 1
        assert 1 in report.duplicate_indices  # second one removed

    def test_no_duplicates(self):
        """Well-separated embeddings → no duplicates."""
        embeddings = np.eye(5)
        det = DuplicateDetector(threshold=0.1)
        report = det.detect(embeddings)

        assert report.n_duplicates == 0
        assert len(report.duplicate_indices) == 0

    def test_threshold_sensitivity(self):
        """Higher threshold → more things flagged as duplicates."""
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(20, 4))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms

        strict = DuplicateDetector(threshold=0.01).detect(embeddings)
        loose = DuplicateDetector(threshold=1.5).detect(embeddings)

        assert strict.n_duplicates <= loose.n_duplicates

    def test_distance_matrix_shape(self):
        embeddings = np.random.randn(10, 4)
        report = DuplicateDetector(threshold=0.5).detect(embeddings)
        assert report.distance_matrix.shape == (10, 10)
