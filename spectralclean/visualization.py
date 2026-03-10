"""
Visualisation helpers for SpectralClean.

Produces publication-ready plots for:
- Spectral score distributions (histogram + GMM decision boundary)
- Extreme samples grids (lowest/highest scoring crops)
- Cluster scatter plots (PCA / t-SNE)
- Pairwise distance distributions (for duplicate threshold tuning)
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


# Apply a clean style globally
sns.set_theme(style="whitegrid")


# ======================================================================
# Score Distribution
# ======================================================================

def plot_score_distribution(
    scores: np.ndarray,
    clean_mean: Optional[float] = None,
    noisy_mean: Optional[float] = None,
    n_clean: Optional[int] = None,
    n_noisy: Optional[int] = None,
    title: Optional[str] = None,
    output_path: str = "score_distribution.png",
) -> None:
    """Histogram of spectral scores with optional GMM decision lines."""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, density=True, alpha=0.5, color="gray", label="Score distribution")

    if clean_mean is not None:
        plt.axvline(x=clean_mean, color="g", linestyle="--", linewidth=2, label=f"Clean mean ({clean_mean:.4f})")
    if noisy_mean is not None:
        plt.axvline(x=noisy_mean, color="r", linestyle="--", linewidth=2, label=f"Noisy mean ({noisy_mean:.4f})")

    subtitle_parts = []
    if n_clean is not None:
        subtitle_parts.append(f"{n_clean} clean")
    if n_noisy is not None:
        subtitle_parts.append(f"{n_noisy} noisy")

    plt.title(title or "Spectral Score Distribution")
    if subtitle_parts:
        plt.xlabel(f"Score  ({', '.join(subtitle_parts)})")
    else:
        plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


# ======================================================================
# Extreme Samples Grid
# ======================================================================

def plot_extreme_samples(
    image_paths: List[str],
    scores: np.ndarray,
    output_dir: str,
    n_show: int = 25,
) -> None:
    """Save grids of the lowest- and highest-scoring crops."""
    sorted_indices = np.argsort(scores)
    low_indices = sorted_indices[:n_show]
    high_indices = sorted_indices[-n_show:][::-1]

    _save_grid(
        image_paths, low_indices, scores,
        f"Bottom {n_show} — Likely Noise / Outliers",
        os.path.join(output_dir, "extreme_low_scores.png"),
    )
    _save_grid(
        image_paths, high_indices, scores,
        f"Top {n_show} — Likely Redundant / Dominant",
        os.path.join(output_dir, "extreme_high_scores.png"),
    )


def _save_grid(
    paths: List[str],
    indices: np.ndarray,
    scores: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    if len(indices) == 0:
        return

    n = len(indices)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax in axes_flat:
        ax.axis("off")

    for i, idx in enumerate(indices):
        try:
            img = Image.open(paths[idx]).convert("RGB")
            axes_flat[i].imshow(img)
            axes_flat[i].set_title(f"{scores[idx]:.3f}", fontsize=8)
        except Exception:
            pass

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


# ======================================================================
# Cluster Scatter
# ======================================================================

def plot_cluster_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Embedding Clusters (PCA)",
    output_path: str = "cluster_scatter.png",
) -> None:
    """2-D PCA scatter coloured by cluster label."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels, cmap="tab10", alpha=0.6, s=15,
    )
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


# ======================================================================
# Pairwise Distance Distribution (for duplicate threshold tuning)
# ======================================================================

def plot_distance_distribution(
    distance_matrix: np.ndarray,
    threshold: float,
    output_path: str = "distance_distribution.png",
) -> None:
    """Histogram of upper-triangle pairwise distances."""
    triu = np.triu_indices_from(distance_matrix, k=1)
    dists = distance_matrix[triu]
    # Replace inf with NaN for plotting
    dists = dists[np.isfinite(dists)]

    plt.figure(figsize=(10, 6))
    plt.hist(dists, bins=100, range=(0.0, 0.5), color="steelblue", alpha=0.7, edgecolor="black")
    plt.axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.2f})")
    plt.title("Pairwise Embedding Distance Distribution")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
