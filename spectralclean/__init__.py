"""
SpectralClean — Spectral eigenvector-based noise detection and cleaning
for COCO object detection datasets.

Usage:
    from spectralclean import SpectralCleaner
    cleaner = SpectralCleaner()
    cleaner.clean("annotations.json", "images/", "output/")
"""

from spectralclean.feature_extractor import FeatureExtractor
from spectralclean.spectral_scorer import SpectralScorer
from spectralclean.noise_separator import NoiseSeparator
from spectralclean.duplicate_detector import DuplicateDetector
from spectralclean.smart_masker import SmartMasker
from spectralclean.coco_utils import CocoDataset
from spectralclean.pipeline import SpectralCleaner

__version__ = "0.1.0"

__all__ = [
    "FeatureExtractor",
    "SpectralScorer",
    "NoiseSeparator",
    "DuplicateDetector",
    "SmartMasker",
    "CocoDataset",
    "SpectralCleaner",
]
