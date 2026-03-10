"""
CLIP-based feature extraction for object detection crops.

Extracts L2-normalized embeddings from cropped object instances using a
pretrained CLIP vision encoder.  Designed for batch processing of large
datasets with progress reporting.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel


class FeatureExtractor:
    """Extract normalized CLIP embeddings from a list of image paths.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the CLIP vision encoder.
    device : str or None
        ``"cuda"``, ``"cpu"``, or *None* for auto-detection.
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPVisionModel.from_pretrained(
            model_id, use_safetensors=True
        ).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Return ``(N, D)`` L2-normalised embeddings for *image_paths*.

        Invalid or unreadable images are silently skipped; the returned
        array may therefore have fewer rows than ``len(image_paths)``.
        The caller can use :pymethod:`extract_with_paths` to also get the
        valid paths back.
        """
        embeddings, _ = self.extract_with_paths(
            image_paths, batch_size=batch_size, show_progress=show_progress
        )
        return embeddings

    def extract_with_paths(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, List[str]]:
        """Return ``(embeddings, valid_paths)`` filtering out bad images."""
        all_embeddings: list[np.ndarray] = []
        valid_paths: List[str] = []

        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features", unit="batch")

        for start in iterator:
            batch_paths = image_paths[start : start + batch_size]
            images: list[Image.Image] = []
            good_paths: list[str] = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    good_paths.append(p)
                except Exception:
                    continue

            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(**inputs)

            embeds = outputs.pooler_output
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeds.cpu().numpy())
            valid_paths.extend(good_paths)

        if not all_embeddings:
            return np.empty((0, 0)), []

        return np.vstack(all_embeddings), valid_paths
