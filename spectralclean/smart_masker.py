"""
Overlap-safe annotation masking for COCO datasets.

When noisy annotations are removed from an image, their bounding-box
regions are painted over with a neutral colour (grey).  However, if a
valid annotation *overlaps* the noisy one, those shared pixels must be
protected to avoid corrupting clean labels.

This module implements that "smart masking" logic.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Set

import cv2
import numpy as np
from tqdm import tqdm


class SmartMasker:
    """Mask noisy annotations while protecting overlapping valid ones.

    Parameters
    ----------
    mask_colour : tuple of int
        BGR colour used to paint over masked regions.
    """

    def __init__(self, mask_colour: tuple = (128, 128, 128)) -> None:
        self.mask_colour = mask_colour

    def apply(
        self,
        coco_data: dict,
        ids_to_mask: Set[int],
        input_img_root: str,
        output_img_root: str,
        output_json_path: str,
    ) -> dict:
        """Mask annotations and write the cleaned dataset.

        Parameters
        ----------
        coco_data : dict
            Full COCO-format annotation dict.
        ids_to_mask : set of int
            Annotation IDs to remove (and mask visually).
        input_img_root : str
            Directory containing source images.
        output_img_root : str
            Directory for the output images.
        output_json_path : str
            Path for the cleaned JSON file.

        Returns
        -------
        dict
            The cleaned COCO data (annotations filtered).
        """
        ids_to_mask = set(ids_to_mask)

        # Group annotations by image: keep vs mask
        img_to_anns: Dict[int, Dict[str, list]] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            bucket = img_to_anns.setdefault(img_id, {"keep": [], "mask": []})
            if ann["id"] in ids_to_mask:
                bucket["mask"].append(ann["bbox"])
            else:
                bucket["keep"].append(ann["bbox"])

        os.makedirs(output_img_root, exist_ok=True)

        masked_count = 0
        copied_count = 0

        for img_info in tqdm(coco_data["images"], desc="Smart-masking images"):
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            src = os.path.join(input_img_root, file_name)
            dst = os.path.join(output_img_root, file_name)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

            buckets = img_to_anns.get(img_id)
            if buckets and buckets["mask"]:
                img = cv2.imread(src)
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Protected zone: valid annotations
                safe_mask = np.zeros((h, w), dtype=np.uint8)
                for (bx, by, bw, bh) in buckets["keep"]:
                    bx, by, bw, bh = map(int, [bx, by, bw, bh])
                    cv2.rectangle(safe_mask, (bx, by), (bx + bw, by + bh), 255, -1)

                # Paint zone: noisy annotations
                paint_mask = np.zeros((h, w), dtype=np.uint8)
                for (bx, by, bw, bh) in buckets["mask"]:
                    bx, by, bw, bh = map(int, [bx, by, bw, bh])
                    cv2.rectangle(paint_mask, (bx, by), (bx + bw, by + bh), 255, -1)

                # Subtract safe zone from paint zone
                final_mask = cv2.bitwise_and(paint_mask, cv2.bitwise_not(safe_mask))
                img[final_mask > 0] = self.mask_colour

                cv2.imwrite(dst, img)
                masked_count += 1
            else:
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_count += 1

        # Filter annotations
        import copy

        cleaned = copy.deepcopy(coco_data)
        cleaned["annotations"] = [
            ann for ann in cleaned["annotations"] if ann["id"] not in ids_to_mask
        ]

        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(cleaned, f)

        return cleaned
