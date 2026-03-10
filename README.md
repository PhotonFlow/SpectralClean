<p align="center">
  <h1 align="center">🔬 SpectralClean</h1>
  <p align="center"><strong>Spectral eigenvector-based noise detection and cleaning for COCO object detection datasets</strong></p>
  <p align="center">
    <a href="#installation">Install</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#cli-reference">CLI</a> •
    <a href="#python-api">API</a> •
    <a href="#citation">Citation</a>
  </p>
</p>

---

**SpectralClean** automatically detects and removes **noisy annotations** and **semantic duplicates** from COCO-format object detection datasets — without requiring any manual inspection.

It uses **spectral analysis of CLIP embeddings**: by projecting per-instance feature vectors onto the dominant eigenvectors of their Gram matrix, it assigns a "typicality score" to every annotation.  A Gaussian Mixture Model then separates clean samples from noise.  An optional **smart masking** step paints over removed regions while protecting overlapping valid annotations.

## ✨ Key Features

| Feature | Description |
|---|---|
| **Spectral Noise Scoring** | Top-K eigenvector subspace projection on CLIP embeddings — extends the FINE algorithm from classification to object detection |
| **GMM Separation** | Two-component Gaussian Mixture Model automatically finds the clean/noisy decision boundary |
| **Semantic Deduplication** | Flags near-duplicate instances using pairwise embedding distance |
| **Smart Masking** | Masks removed annotations with grey fill while *protecting* pixels of overlapping valid annotations |
| **Intra-Image Dedup** | Removes duplicate bounding boxes on the same image (same class, high IoU) |
| **Publication-Ready Plots** | Score distributions, extreme-sample grids, distance histograms, cluster scatter plots |
| **CLI + Python API** | Use from the command line or import as a library |

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SpectralClean Pipeline                         │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐              │
│  │ COCO JSON │──▶│ Crop Object  │──▶│ CLIP Feature  │              │
│  │ + Images  │   │ Instances    │   │ Extraction    │              │
│  └──────────┘   └──────────────┘   └───────┬───────┘              │
│                                             │                       │
│                                     (N, 512) embeddings             │
│                                             │                       │
│                    ┌────────────────────────┼───────────────┐       │
│                    │                        │               │       │
│              ┌─────▼──────┐          ┌──────▼─────┐  ┌──────▼────┐ │
│              │ Spectral   │          │ Semantic   │  │ Intra-Img │ │
│              │ Scoring    │          │ Dedup      │  │ Dedup     │ │
│              │ (Gram →    │          │ (Pairwise  │  │ (IoU)     │ │
│              │  Eigen →   │          │  Distance) │  │           │ │
│              │  Project)  │          └──────┬─────┘  └──────┬────┘ │
│              └─────┬──────┘                 │               │       │
│                    │                        │               │       │
│              ┌─────▼──────┐                 │               │       │
│              │ GMM        │                 │               │       │
│              │ Separation │                 │               │       │
│              └─────┬──────┘                 │               │       │
│                    │                        │               │       │
│                    └────────────┬───────────┘               │       │
│                                │                            │       │
│                         IDs to Remove ◄─────────────────────┘       │
│                                │                                    │
│                         ┌──────▼──────┐                             │
│                         │ Smart Mask  │                             │
│                         │ (Overlap-   │                             │
│                         │  Safe)      │                             │
│                         └──────┬──────┘                             │
│                                │                                    │
│                    ┌───────────┴───────────┐                        │
│                    │                       │                        │
│             ┌──────▼──────┐         ┌──────▼──────┐                │
│             │ Clean JSON  │         │ Masked      │                │
│             │ (filtered)  │         │ Images      │                │
│             └─────────────┘         └─────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### The Math Behind It

1. **Feature Extraction**: Each cropped object instance is embedded using a pretrained CLIP vision encoder → L2-normalised vector in ℝ<sup>512</sup>

2. **Gram Matrix**: Compute **G = Eᵀ E** where E is the (N × D) embedding matrix

3. **Eigen-decomposition**: Solve **G v = λ v** to find the principal directions of the embedding space

4. **Projection & Scoring**: Project each embedding onto the top-K eigenvectors and compute the squared projection norm:
   ```
   score(xᵢ) = ‖ Eᵢ · V_K ‖²
   ```
   - **Low scores** → the sample is far from the dominant subspace → likely noise or outlier
   - **High scores** → the sample is overly aligned → likely redundant or a dominant prototype

5. **GMM Separation**: Fit a 2-component GMM to the score distribution; the component with lower mean = clean data

## Installation

```bash
# From source
git clone https://github.com/alanpeng/SpectralClean.git
cd SpectralClean
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- transformers, scikit-learn, opencv-python, matplotlib, click

## Quickstart

### CLI

```bash
# Full cleaning pipeline
spectralclean clean annotations.json images/ -o cleaned/ -c person vehicle

# Analysis only (no file modifications)
spectralclean analyze annotations.json images/ -o analysis/

# Deduplication only
spectralclean deduplicate annotations.json images/ --threshold 0.12
```

### Python API

```python
from spectralclean import SpectralCleaner

cleaner = SpectralCleaner(
    top_k=4,              # eigenvectors to use
    gmm_threshold=0.45,   # clean probability cutoff
    dedup_threshold=0.15,  # embedding distance for dedup
)

summary = cleaner.clean(
    json_path="annotations.json",
    image_root="images/",
    output_dir="cleaned_output/",
    target_classes=["person", "vehicle"],
)

print(summary)
# {'person': {'total': 5000, 'clean': 4200, 'noisy': 800, ...}, ...}
```

### Modular Usage

```python
from spectralclean import FeatureExtractor, SpectralScorer, NoiseSeparator

# Step 1: Extract features
extractor = FeatureExtractor()
embeddings = extractor.extract(image_paths)

# Step 2: Compute spectral scores
scorer = SpectralScorer(top_k=4)
result = scorer.score(embeddings)

# Step 3: Separate clean from noisy
separator = NoiseSeparator(threshold=0.45)
sep = separator.separate(result.scores)

print(f"Clean: {len(sep.clean_indices)}, Noisy: {len(sep.noisy_indices)}")
```

## CLI Reference

| Command | Description |
|---|---|
| `spectralclean clean` | Full pipeline: score → separate → dedup → mask → output |
| `spectralclean analyze` | Score and visualise without modifying files |
| `spectralclean deduplicate` | Find and remove semantic duplicates only |

### Common Options

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `spectralclean_output` | Output directory |
| `-c, --classes` | all | Target classes |
| `-k, --top-k` | 4 | Number of eigenvectors |
| `--gmm-threshold` | 0.45 | Clean probability cutoff |
| `--dedup-threshold` | 0.15 | Dedup distance threshold |
| `--no-dedup` | false | Skip deduplication |
| `--no-viz` | false | Skip visualisations |
| `--device` | auto | `cuda` / `cpu` |

## Output Structure

```
spectralclean_output/
├── images/                      # Cleaned images (noisy regions masked)
├── annotations_clean.json       # Filtered COCO JSON
├── reports/
│   ├── cleaning_report.txt      # Summary statistics
│   ├── person/
│   │   ├── score_distribution.png
│   │   ├── extreme_low_scores.png
│   │   ├── extreme_high_scores.png
│   │   └── distance_distribution.png
│   └── vehicle/
│       └── ...
└── _temp_crops/                 # Intermediate crops (can be deleted)
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## How Is This Different from FINE?

[FINE (Zhu et al., 2022)](https://arxiv.org/abs/2110.06283) was designed for **image classification** and uses a **single** eigenvector.  SpectralClean extends it:

| | FINE | SpectralClean |
|---|---|---|
| **Task** | Classification | Object Detection |
| **Eigenvectors** | 1 (first) | Top-K subspace |
| **Granularity** | Image-level | Annotation-level |
| **Masking** | Remove image | Smart mask (overlap-safe) |
| **Deduplication** | ✗ | ✓ (embedding-distance) |
| **Intra-image dedup** | ✗ | ✓ (IoU-based) |
| **Visualisation** | Basic | Score distributions, extreme grids, distance plots |

## Citation

If you find SpectralClean useful, please consider citing:

```bibtex
@software{peng2026spectralclean,
  author = {Alan Peng},
  title = {SpectralClean: Spectral Eigenvector-Based Noise Detection for Object Detection Datasets},
  year = {2026},
  url = {https://github.com/alanpeng/SpectralClean}
}
```

## License

MIT — see [LICENSE](LICENSE).
