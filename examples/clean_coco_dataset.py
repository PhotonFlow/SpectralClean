"""
Example: Clean a COCO object detection dataset using SpectralClean.

Usage:
    python examples/clean_coco_dataset.py \\
        --json data/annotations.json \\
        --images data/train/ \\
        --output cleaned_output/ \\
        --classes person vehicle
"""

import argparse

from spectralclean import SpectralCleaner


def main():
    parser = argparse.ArgumentParser(
        description="Clean a COCO dataset using spectral noise filtering."
    )
    parser.add_argument(
        "--json", required=True, help="Path to COCO annotation JSON file."
    )
    parser.add_argument(
        "--images", required=True, help="Directory containing source images."
    )
    parser.add_argument(
        "--output",
        default="spectralclean_output",
        help="Output directory (default: spectralclean_output).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Target classes to clean (default: all classes).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of leading eigenvectors (default: 4).",
    )
    parser.add_argument(
        "--gmm-threshold",
        type=float,
        default=0.45,
        help="GMM clean-probability threshold (default: 0.45).",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.15,
        help="Embedding distance for dedup (default: 0.15).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip semantic deduplication.",
    )
    args = parser.parse_args()

    # Initialize the cleaner
    cleaner = SpectralCleaner(
        top_k=args.top_k,
        gmm_threshold=args.gmm_threshold,
        dedup_threshold=args.dedup_threshold,
    )

    # Run the pipeline
    summary = cleaner.clean(
        json_path=args.json,
        image_root=args.images,
        output_dir=args.output,
        target_classes=args.classes,
        run_dedup=not args.no_dedup,
        visualize=True,
    )

    # Print results
    print("\n" + "=" * 50)
    print("  CLEANING SUMMARY")
    print("=" * 50)
    for cls, stats in summary.items():
        print(f"\n  {cls}:")
        print(f"    Total:  {stats['total']}")
        print(f"    Clean:  {stats['clean']}")
        print(f"    Noisy:  {stats['noisy']}")
        if "duplicates" in stats:
            print(f"    Dupes:  {stats['duplicates']}")


if __name__ == "__main__":
    main()
