"""
COCO format dataset utilities.

Provides helpers for loading COCO JSON annotations, cropping object
instances from images, computing IoU, and removing intra-image duplicate
annotations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ======================================================================
# IoU
# ======================================================================

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two COCO-format boxes ``[x, y, w, h]``."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    b1_x2, b1_y2 = x1 + w1, y1 + h1
    b2_x2, b2_y2 = x2 + w2, y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union if union > 0 else 0.0


# ======================================================================
# COCO Dataset Wrapper
# ======================================================================

@dataclass
class CropInfo:
    """Metadata for a single cropped instance."""

    annotation_id: int
    image_id: int
    category_id: int
    category_name: str
    bbox: List[float]
    crop_path: str


class CocoDataset:
    """Lightweight COCO JSON loader with crop and dedup utilities.

    Parameters
    ----------
    json_path : str
        Path to a COCO-format annotation JSON.
    image_root : str
        Directory containing the source images.
    """

    def __init__(self, json_path: str, image_root: str) -> None:
        self.json_path = json_path
        self.image_root = Path(image_root)
        self._data: dict = {}
        self._image_map: Dict[int, str] = {}
        self._category_map: Dict[int, str] = {}
        self._load()

    @property
    def annotations(self) -> list:
        return self._data.get("annotations", [])

    @property
    def images(self) -> list:
        return self._data.get("images", [])

    @property
    def categories(self) -> list:
        return self._data.get("categories", [])

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with open(self.json_path, "r") as f:
            self._data = json.load(f)
        self._image_map = {
            img["id"]: img["file_name"] for img in self._data["images"]
        }
        self._category_map = {
            cat["id"]: cat["name"] for cat in self._data["categories"]
        }

    # ------------------------------------------------------------------
    # Cropping
    # ------------------------------------------------------------------

    def crop_instances(
        self,
        target_class: str,
        output_dir: str,
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[CropInfo]:
        """Crop all instances of *target_class* and save to *output_dir*.

        Returns a list of :class:`CropInfo` for successfully cropped
        instances.
        """
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        target_anns = [
            ann
            for ann in self.annotations
            if self._category_map.get(ann["category_id"]) == target_class
        ]

        if limit is not None and limit < len(target_anns):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(target_anns), size=limit, replace=False)
            target_anns = [target_anns[i] for i in indices]

        results: List[CropInfo] = []
        iterator = tqdm(target_anns, desc=f"Cropping '{target_class}'") if show_progress else target_anns

        for ann in iterator:
            img_id = ann["image_id"]
            file_name = self._image_map.get(img_id)
            if file_name is None:
                continue

            img_path = self.image_root / file_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            x, y, w, h = map(int, ann["bbox"])
            h_img, w_img = img.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{ann['id']}_{file_name}"
            out_path = save_dir / out_name
            cv2.imwrite(str(out_path), crop)

            results.append(
                CropInfo(
                    annotation_id=ann["id"],
                    image_id=img_id,
                    category_id=ann["category_id"],
                    category_name=target_class,
                    bbox=ann["bbox"],
                    crop_path=str(out_path),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Intra-image duplicate removal
    # ------------------------------------------------------------------

    def remove_intra_duplicates(
        self,
        iou_threshold: float = 0.85,
        output_json: Optional[str] = None,
    ) -> Tuple[dict, Set[int]]:
        """Remove duplicate annotations within the same image.

        Two annotations on the same image with the same category and
        IoU >= *iou_threshold* are considered duplicates.  The smaller
        (by area) annotation is removed.

        Returns
        -------
        cleaned_data : dict
            A copy of the COCO data with duplicates removed.
        removed_ids : set of int
            Annotation IDs that were removed.
        """
        img_to_anns: Dict[int, list] = {}
        for ann in self.annotations:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

        ids_to_remove: Set[int] = set()

        for img_id, anns in tqdm(
            img_to_anns.items(), desc="Checking intra-image duplicates"
        ):
            if len(anns) <= 1:
                continue

            # Sort by area descending — keep larger boxes
            anns.sort(
                key=lambda a: a["bbox"][2] * a["bbox"][3], reverse=True
            )
            kept_indices: List[int] = []

            for i, curr in enumerate(anns):
                is_dup = False
                for ki in kept_indices:
                    kept = anns[ki]
                    if curr["category_id"] != kept["category_id"]:
                        continue
                    if compute_iou(curr["bbox"], kept["bbox"]) >= iou_threshold:
                        is_dup = True
                        break

                if is_dup:
                    ids_to_remove.add(curr["id"])
                else:
                    kept_indices.append(i)

        # Build cleaned copy
        import copy

        cleaned = copy.deepcopy(self._data)
        cleaned["annotations"] = [
            ann for ann in cleaned["annotations"] if ann["id"] not in ids_to_remove
        ]

        if output_json:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, "w") as f:
                json.dump(cleaned, f)

        return cleaned, ids_to_remove

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_json(data: dict, path: str) -> None:
        """Write a COCO dict to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
