import argparse
import json
from pathlib import Path
from typing import Tuple

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.common.structured_sampling_index import pairwise_l2_knn


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.mean(x, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    comps = vh[:2].T
    return x @ comps


def kmeans(
    x: np.ndarray,
    k: int,
    seed: int = 42,
    max_iters: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, n)
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iters):
        d = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centroids[c] = np.mean(x[mask], axis=0)
            else:
                centroids[c] = x[rng.integers(0, n)]
    return labels, centroids


def main():
    parser = argparse.ArgumentParser(description="Analyze wrist embedding KNN clustering quality.")
    parser.add_argument("--wrist_embeddings_path", type=str, required=True, help="Path to wrist embeddings .npy")
    parser.add_argument("--output_dir", type=str, default=".cache/structured_sampling/analysis", help="Output dir")
    parser.add_argument("--sample_n", type=int, default=5000, help="Max number of samples to analyze")
    parser.add_argument("--k", type=int, default=16, help="KNN neighbors")
    parser.add_argument("--kmeans_k", type=int, default=8, help="KMeans cluster count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk_size", type=int, default=2048, help="KNN chunk size")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.wrist_embeddings_path)
    if emb.ndim != 2:
        raise ValueError(f"Expected [N, D] embeddings, got {emb.shape}")
    n = emb.shape[0]

    rng = np.random.default_rng(args.seed)
    if args.sample_n < n:
        sub_idx = np.sort(rng.choice(n, size=args.sample_n, replace=False))
        x = emb[sub_idx]
    else:
        sub_idx = np.arange(n, dtype=np.int64)
        x = emb

    if len(x) < 3:
        raise ValueError("Need at least 3 samples for clustering analysis.")

    p2 = pca_2d(x)
    labels, centroids = kmeans(p2, k=args.kmeans_k, seed=args.seed)

    knn_idx, knn_dist = pairwise_l2_knn(x, k=args.k, exclude_self=True, chunk_size=args.chunk_size)
    same_cluster = []
    for i in range(len(x)):
        neigh = knn_idx[i]
        valid = neigh[neigh >= 0]
        if len(valid) == 0:
            continue
        same_cluster.append(np.mean(labels[valid] == labels[i]))
    same_cluster_ratio = float(np.mean(same_cluster)) if len(same_cluster) else 0.0

    # Scatter with cluster colors.
    plt.figure(figsize=(8, 7))
    plt.scatter(p2[:, 0], p2[:, 1], c=labels, s=5, cmap="tab20", alpha=0.75)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", s=40, marker="x")
    plt.title("Wrist Embedding PCA (colored by KMeans cluster)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_dir / "wrist_pca_clusters.png", dpi=180)
    plt.close()

    # KNN distance histogram.
    plt.figure(figsize=(8, 5))
    valid_dist = knn_dist[np.isfinite(knn_dist)]
    plt.hist(valid_dist, bins=60, alpha=0.85)
    plt.title("Wrist KNN Distance Histogram")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "wrist_knn_distance_hist.png", dpi=180)
    plt.close()

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}
    report = {
        "num_total_samples": int(n),
        "num_analyzed_samples": int(len(x)),
        "embedding_dim": int(emb.shape[1]),
        "k": int(args.k),
        "kmeans_k": int(args.kmeans_k),
        "same_cluster_ratio_mean": same_cluster_ratio,
        "knn_distance_mean": float(np.mean(valid_dist)) if len(valid_dist) else None,
        "knn_distance_std": float(np.std(valid_dist)) if len(valid_dist) else None,
        "cluster_sizes": cluster_sizes,
        "artifacts": {
            "pca_clusters": str((out_dir / "wrist_pca_clusters.png").resolve()),
            "knn_hist": str((out_dir / "wrist_knn_distance_hist.png").resolve()),
        },
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True, sort_keys=True)

    print("Analysis done.")
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
