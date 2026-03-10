"""
Command-line interface for SpectralClean.

Usage examples::

    # Full cleaning pipeline
    spectralclean clean annotations.json images/ --output cleaned/ --classes person vehicle

    # Analyse a dataset (produce reports only, no masking)
    spectralclean analyze annotations.json images/ --output analysis/

    # Deduplicate only
    spectralclean deduplicate annotations.json images/ --output deduped/ --threshold 0.15
"""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="spectralclean")
def main() -> None:
    """SpectralClean — spectral noise detection for COCO datasets."""


# ======================================================================
# clean
# ======================================================================

@main.command()
@click.argument("json_path", type=click.Path(exists=True))
@click.argument("image_root", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_dir",
    default="spectralclean_output",
    show_default=True,
    help="Output directory for cleaned dataset and reports.",
)
@click.option(
    "-c", "--classes",
    multiple=True,
    help="Target classes to analyse (default: all).",
)
@click.option(
    "-k", "--top-k",
    default=4,
    show_default=True,
    help="Number of top eigenvectors for spectral scoring.",
)
@click.option(
    "--gmm-threshold",
    default=0.45,
    show_default=True,
    help="GMM probability threshold for clean/noisy separation.",
)
@click.option(
    "--dedup-threshold",
    default=0.15,
    show_default=True,
    help="Embedding distance threshold for deduplication.",
)
@click.option(
    "--no-dedup",
    is_flag=True,
    default=False,
    help="Skip semantic deduplication.",
)
@click.option(
    "--no-viz",
    is_flag=True,
    default=False,
    help="Skip visualisation plots.",
)
@click.option(
    "--device",
    default=None,
    help='Compute device ("cuda" / "cpu" / auto).',
)
def clean(
    json_path: str,
    image_root: str,
    output_dir: str,
    classes: tuple,
    top_k: int,
    gmm_threshold: float,
    dedup_threshold: float,
    no_dedup: bool,
    no_viz: bool,
    device: str | None,
) -> None:
    """Run the full spectral cleaning pipeline on a COCO dataset."""
    from spectralclean.pipeline import SpectralCleaner

    cleaner = SpectralCleaner(
        top_k=top_k,
        gmm_threshold=gmm_threshold,
        dedup_threshold=dedup_threshold,
        device=device,
    )

    target_classes = list(classes) if classes else None

    summary = cleaner.clean(
        json_path=json_path,
        image_root=image_root,
        output_dir=output_dir,
        target_classes=target_classes,
        run_dedup=not no_dedup,
        visualize=not no_viz,
    )

    click.echo("\n✓ Cleaning complete.")
    for cls_name, stats in summary.items():
        click.echo(
            f"  {cls_name}: {stats['clean']} clean / {stats['noisy']} noisy"
        )


# ======================================================================
# analyze (reports only, no masking)
# ======================================================================

@main.command()
@click.argument("json_path", type=click.Path(exists=True))
@click.argument("image_root", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_dir",
    default="spectralclean_analysis",
    show_default=True,
    help="Output directory for analysis reports.",
)
@click.option(
    "-c", "--classes",
    multiple=True,
    help="Target classes to analyse (default: all).",
)
@click.option(
    "-k", "--top-k",
    default=4,
    show_default=True,
    help="Number of top eigenvectors.",
)
@click.option(
    "--device",
    default=None,
    help='Compute device ("cuda" / "cpu" / auto).',
)
def analyze(
    json_path: str,
    image_root: str,
    output_dir: str,
    classes: tuple,
    top_k: int,
    device: str | None,
) -> None:
    """Analyse dataset quality without modifying anything."""
    import json
    import os

    from spectralclean.coco_utils import CocoDataset
    from spectralclean.feature_extractor import FeatureExtractor
    from spectralclean.noise_separator import NoiseSeparator
    from spectralclean.spectral_scorer import SpectralScorer
    from spectralclean import visualization as viz

    dataset = CocoDataset(json_path, image_root)
    extractor = FeatureExtractor(device=device)
    scorer = SpectralScorer(top_k=top_k)
    separator = NoiseSeparator()

    target_classes = list(classes) if classes else [
        cat["name"] for cat in dataset.categories
    ]

    crop_dir = os.path.join(output_dir, "_temp_crops")
    os.makedirs(output_dir, exist_ok=True)

    for cls_name in target_classes:
        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        click.echo(f"\nAnalysing: {cls_name}")

        crop_infos = dataset.crop_instances(
            target_class=cls_name,
            output_dir=os.path.join(crop_dir, cls_name),
        )
        if not crop_infos:
            click.echo(f"  No instances for '{cls_name}'.")
            continue

        crop_paths = [ci.crop_path for ci in crop_infos]
        embeddings, valid_paths = extractor.extract_with_paths(crop_paths)
        if len(embeddings) == 0:
            continue

        result = scorer.score(embeddings)
        sep = separator.separate(result.scores)

        viz.plot_score_distribution(
            result.scores,
            clean_mean=sep.clean_mean,
            noisy_mean=sep.noisy_mean,
            n_clean=len(sep.clean_indices),
            n_noisy=len(sep.noisy_indices),
            title=f"Spectral Score — {cls_name}",
            output_path=os.path.join(cls_dir, "score_distribution.png"),
        )
        viz.plot_extreme_samples(valid_paths, result.scores, cls_dir, n_show=25)

        click.echo(
            f"  {len(sep.clean_indices)} clean / {len(sep.noisy_indices)} noisy"
        )

    click.echo(f"\n✓ Analysis complete. See: {output_dir}")


# ======================================================================
# deduplicate
# ======================================================================

@main.command()
@click.argument("json_path", type=click.Path(exists=True))
@click.argument("image_root", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_dir",
    default="spectralclean_deduped",
    show_default=True,
    help="Output directory for deduplicated dataset.",
)
@click.option(
    "-c", "--classes",
    multiple=True,
    help="Target classes to check (default: all).",
)
@click.option(
    "--threshold",
    default=0.15,
    show_default=True,
    help="Embedding distance threshold for duplicates.",
)
@click.option(
    "--device",
    default=None,
    help='Compute device ("cuda" / "cpu" / auto).',
)
def deduplicate(
    json_path: str,
    image_root: str,
    output_dir: str,
    classes: tuple,
    threshold: float,
    device: str | None,
) -> None:
    """Find and remove semantic duplicates from a COCO dataset."""
    import json
    import os

    from spectralclean.coco_utils import CocoDataset
    from spectralclean.duplicate_detector import DuplicateDetector
    from spectralclean.feature_extractor import FeatureExtractor
    from spectralclean.smart_masker import SmartMasker
    from spectralclean import visualization as viz

    dataset = CocoDataset(json_path, image_root)
    extractor = FeatureExtractor(device=device)
    detector = DuplicateDetector(threshold=threshold)
    masker = SmartMasker()

    target_classes = list(classes) if classes else [
        cat["name"] for cat in dataset.categories
    ]

    crop_dir = os.path.join(output_dir, "_temp_crops")
    viz_dir = os.path.join(output_dir, "reports")
    os.makedirs(viz_dir, exist_ok=True)

    all_ids: set = set()

    for cls_name in target_classes:
        click.echo(f"\nDeduplicating: {cls_name}")
        crop_infos = dataset.crop_instances(
            target_class=cls_name,
            output_dir=os.path.join(crop_dir, cls_name),
        )
        if not crop_infos:
            continue

        crop_paths = [ci.crop_path for ci in crop_infos]
        embeddings, valid_paths = extractor.extract_with_paths(crop_paths)
        if len(embeddings) == 0:
            continue

        path_to_id = {ci.crop_path: ci.annotation_id for ci in crop_infos}
        valid_ids = [path_to_id[p] for p in valid_paths]

        report = detector.detect(embeddings)
        dup_ids = {valid_ids[i] for i in report.duplicate_indices}
        all_ids.update(dup_ids)

        click.echo(f"  Found {report.n_duplicates} duplicates.")

        viz.plot_distance_distribution(
            report.distance_matrix,
            threshold,
            os.path.join(viz_dir, f"{cls_name}_distance_dist.png"),
        )

    click.echo(f"\nTotal duplicates: {len(all_ids)}")

    if all_ids:
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        masker.apply(
            coco_data=coco_data,
            ids_to_mask=all_ids,
            input_img_root=image_root,
            output_img_root=os.path.join(output_dir, "images"),
            output_json_path=os.path.join(output_dir, "annotations_deduped.json"),
        )

    click.echo(f"✓ Deduplication complete. See: {output_dir}")


if __name__ == "__main__":
    main()
