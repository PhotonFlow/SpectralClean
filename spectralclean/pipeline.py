"""
End-to-end pipeline orchestrator for SpectralClean.

Ties together feature extraction, spectral scoring, noise separation,
duplicate detection, smart masking, and visualisation into a single
high-level API.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Set

import numpy as np

from spectralclean.coco_utils import CocoDataset, CropInfo
from spectralclean.duplicate_detector import DuplicateDetector
from spectralclean.feature_extractor import FeatureExtractor
from spectralclean.noise_separator import NoiseSeparator
from spectralclean.smart_masker import SmartMasker
from spectralclean.spectral_scorer import SpectralScorer
from spectralclean import visualization as viz


class SpectralCleaner:
    """One-call interface for cleaning a COCO detection dataset.

    Parameters
    ----------
    top_k : int
        Number of leading eigenvectors for spectral scoring.
    gmm_threshold : float
        Probability threshold for the GMM clean/noisy split.
    dedup_threshold : float
        Embedding distance below which two crops are duplicates.
    clip_model : str
        HuggingFace CLIP model ID for feature extraction.
    device : str or None
        Compute device (``"cuda"`` / ``"cpu"`` / auto).
    """

    def __init__(
        self,
        top_k: int = 4,
        gmm_threshold: float = 0.45,
        dedup_threshold: float = 0.15,
        clip_model: str = FeatureExtractor.DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.extractor = FeatureExtractor(model_id=clip_model, device=device)
        self.scorer = SpectralScorer(top_k=top_k)
        self.separator = NoiseSeparator(threshold=gmm_threshold)
        self.deduplicator = DuplicateDetector(threshold=dedup_threshold)
        self.masker = SmartMasker()

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def clean(
        self,
        json_path: str,
        image_root: str,
        output_dir: str,
        target_classes: Optional[List[str]] = None,
        run_dedup: bool = True,
        visualize: bool = True,
    ) -> Dict[str, object]:
        """Run the full cleaning pipeline.

        Parameters
        ----------
        json_path : str
            Path to the COCO-format annotation JSON.
        image_root : str
            Directory of source images.
        output_dir : str
            Directory for cleaned dataset and reports.
        target_classes : list of str or None
            Classes to analyse.  *None* → all classes in the dataset.
        run_dedup : bool
            Whether to also run semantic deduplication.
        visualize : bool
            Whether to produce diagnostic plots.

        Returns
        -------
        dict
            Summary statistics keyed by class name.
        """
        dataset = CocoDataset(json_path, image_root)

        if target_classes is None:
            target_classes = list(
                {cat["name"] for cat in dataset.categories}
            )

        crop_dir = os.path.join(output_dir, "_temp_crops")
        viz_dir = os.path.join(output_dir, "reports")
        out_img_dir = os.path.join(output_dir, "images")
        out_json = os.path.join(output_dir, "annotations_clean.json")
        os.makedirs(viz_dir, exist_ok=True)

        all_ids_to_mask: Set[int] = set()
        summary: Dict[str, dict] = {}

        for cls_name in target_classes:
            cls_viz_dir = os.path.join(viz_dir, cls_name)
            os.makedirs(cls_viz_dir, exist_ok=True)

            print(f"\n{'='*50}")
            print(f"  Processing class: {cls_name}")
            print(f"{'='*50}")

            # 1. Crop
            crop_infos = dataset.crop_instances(
                target_class=cls_name,
                output_dir=os.path.join(crop_dir, cls_name),
            )
            if not crop_infos:
                print(f"  No instances found for '{cls_name}'. Skipping.")
                continue

            crop_paths = [ci.crop_path for ci in crop_infos]
            ann_ids = [ci.annotation_id for ci in crop_infos]
            print(f"  Cropped {len(crop_paths)} instances.")

            # 2. Extract features
            embeddings, valid_paths = self.extractor.extract_with_paths(crop_paths)
            if len(embeddings) == 0:
                continue

            # Map valid paths back to annotation IDs
            path_to_id = {ci.crop_path: ci.annotation_id for ci in crop_infos}
            valid_ids = [path_to_id[p] for p in valid_paths]

            # 3. Spectral scoring
            result = self.scorer.score(embeddings)
            print(f"  Score range: [{result.scores.min():.4f}, {result.scores.max():.4f}]")

            # 4. GMM separation
            sep = self.separator.separate(result.scores)
            noisy_ann_ids = {valid_ids[i] for i in sep.noisy_indices}
            all_ids_to_mask.update(noisy_ann_ids)

            cls_summary = {
                "total": len(valid_paths),
                "clean": len(sep.clean_indices),
                "noisy": len(sep.noisy_indices),
                "clean_mean": sep.clean_mean,
                "noisy_mean": sep.noisy_mean,
            }

            print(f"  GMM result: {cls_summary['clean']} clean, {cls_summary['noisy']} noisy")

            # 5. Dedup (optional)
            if run_dedup:
                dedup = self.deduplicator.detect(embeddings)
                dedup_ids = {valid_ids[i] for i in dedup.duplicate_indices}
                all_ids_to_mask.update(dedup_ids)
                cls_summary["duplicates"] = dedup.n_duplicates
                print(f"  Duplicates: {dedup.n_duplicates}")

                if visualize:
                    viz.plot_distance_distribution(
                        dedup.distance_matrix,
                        self.deduplicator.threshold,
                        os.path.join(cls_viz_dir, "distance_distribution.png"),
                    )

            # 6. Visualisation
            if visualize:
                viz.plot_score_distribution(
                    result.scores,
                    clean_mean=sep.clean_mean,
                    noisy_mean=sep.noisy_mean,
                    n_clean=len(sep.clean_indices),
                    n_noisy=len(sep.noisy_indices),
                    title=f"Spectral Score — {cls_name}",
                    output_path=os.path.join(cls_viz_dir, "score_distribution.png"),
                )
                viz.plot_extreme_samples(
                    valid_paths,
                    result.scores,
                    cls_viz_dir,
                    n_show=25,
                )

            summary[cls_name] = cls_summary

        # 7. Intra-image dedup pass
        print(f"\nRunning intra-image duplicate detection...")
        _, intra_removed = dataset.remove_intra_duplicates()
        all_ids_to_mask.update(intra_removed)
        print(f"  Intra-image duplicates: {len(intra_removed)}")

        # 8. Smart masking
        total_to_remove = len(all_ids_to_mask)
        print(f"\nTotal annotations to remove: {total_to_remove}")

        if total_to_remove > 0:
            with open(json_path, "r") as f:
                coco_data = json.load(f)

            self.masker.apply(
                coco_data=coco_data,
                ids_to_mask=all_ids_to_mask,
                input_img_root=image_root,
                output_img_root=out_img_dir,
                output_json_path=out_json,
            )
            print(f"Cleaned dataset saved to: {output_dir}")
        else:
            print("No noisy annotations found. Dataset is clean!")

        # 9. Write summary report
        report_path = os.path.join(viz_dir, "cleaning_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("  SpectralClean — Cleaning Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input:  {json_path}\n")
            f.write(f"Output: {out_json}\n\n")
            for cls_name, stats in summary.items():
                f.write(f"Class: {cls_name}\n")
                for k, v in stats.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
            f.write(f"Intra-image duplicates removed: {len(intra_removed)}\n")
            f.write(f"Total annotations removed: {total_to_remove}\n")

        print(f"Report saved to: {report_path}")
        return summary
