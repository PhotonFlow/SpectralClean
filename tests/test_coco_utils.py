"""Tests for spectralclean.coco_utils."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from spectralclean.coco_utils import CocoDataset, compute_iou


# ======================================================================
# IoU Tests
# ======================================================================

class TestComputeIoU:
    """Unit tests for compute_iou (COCO xywh format)."""

    def test_identical_boxes(self):
        box = [10, 20, 100, 200]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping(self):
        box1 = [0, 0, 10, 10]
        box2 = [100, 100, 10, 10]
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box1 = [0, 0, 10, 10]  # area = 100
        box2 = [5, 5, 10, 10]  # area = 100
        # intersection: x=[5,10], y=[5,10] -> 5*5=25
        # union: 100+100-25 = 175
        expected = 25.0 / 175.0
        assert compute_iou(box1, box2) == pytest.approx(expected, rel=1e-4)

    def test_containment(self):
        outer = [0, 0, 100, 100]  # area = 10000
        inner = [25, 25, 50, 50]  # area = 2500
        # intersection = inner area = 2500
        # union = 10000
        expected = 2500.0 / 10000.0
        assert compute_iou(outer, inner) == pytest.approx(expected, rel=1e-4)

    def test_zero_area(self):
        box1 = [0, 0, 0, 0]
        box2 = [0, 0, 10, 10]
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_touching_edges(self):
        box1 = [0, 0, 10, 10]
        box2 = [10, 0, 10, 10]
        assert compute_iou(box1, box2) == pytest.approx(0.0)


# ======================================================================
# CocoDataset Tests
# ======================================================================

def _make_test_coco(tmp_dir: str) -> str:
    """Create a minimal COCO dataset on disk for testing."""
    img_dir = os.path.join(tmp_dir, "images")
    os.makedirs(img_dir)

    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255  # white square
    cv2.imwrite(os.path.join(img_dir, "test.jpg"), img)

    coco = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 20, 60, 60]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [22, 22, 58, 58]},  # near-dup
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},      # unique
        ],
    }

    json_path = os.path.join(tmp_dir, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(coco, f)

    return json_path, img_dir


class TestCocoDataset:

    def test_load_and_properties(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path, img_dir = _make_test_coco(tmp)
            ds = CocoDataset(json_path, img_dir)

            assert len(ds.annotations) == 3
            assert len(ds.images) == 1
            assert len(ds.categories) == 1

    def test_crop_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path, img_dir = _make_test_coco(tmp)
            ds = CocoDataset(json_path, img_dir)

            crop_dir = os.path.join(tmp, "crops")
            infos = ds.crop_instances("object", crop_dir, show_progress=False)

            assert len(infos) == 3
            for ci in infos:
                assert os.path.exists(ci.crop_path)
                assert ci.category_name == "object"

    def test_crop_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path, img_dir = _make_test_coco(tmp)
            ds = CocoDataset(json_path, img_dir)

            infos = ds.crop_instances("object", os.path.join(tmp, "cr"), limit=1, show_progress=False)
            assert len(infos) == 1

    def test_intra_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path, img_dir = _make_test_coco(tmp)
            ds = CocoDataset(json_path, img_dir)

            cleaned, removed = ds.remove_intra_duplicates(iou_threshold=0.7)

            # Ann 2 (smaller) should be removed as near-dup of Ann 1
            assert 2 in removed
            assert 1 not in removed
            assert 3 not in removed
            assert len(cleaned["annotations"]) == 2
