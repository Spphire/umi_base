from typing import Optional, Tuple

import numpy as np


def compute_future_action_features(
    actions: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Build per-step future-action feature by flattening [t:t+horizon) action slices.
    Tail is padded by repeating the last action.
    """
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D [N, Da], got shape={actions.shape}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    n, da = actions.shape
    out = np.empty((n, horizon * da), dtype=np.float32)
    for t in range(n):
        end = min(n, t + horizon)
        seq = actions[t:end]
        if len(seq) < horizon:
            pad = np.repeat(actions[-1:, :], horizon - len(seq), axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        out[t] = seq.reshape(-1).astype(np.float32)
    return out


def pairwise_l2_knn(
    features: np.ndarray,
    k: int,
    exclude_self: bool = True,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact L2 KNN using chunked matrix distance for offline preprocessing.
    Returns (indices, distances) with shape [N, k].
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [N, D], got shape={features.shape}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    x = features.astype(np.float32, copy=False)
    n, d = x.shape
    if n == 0:
        return np.zeros((0, k), dtype=np.int64), np.zeros((0, k), dtype=np.float32)
    if exclude_self and n == 1:
        return np.full((1, k), -1, dtype=np.int64), np.full((1, k), np.inf, dtype=np.float32)

    max_k = max(1, n - 1) if exclude_self else n
    k_eff = min(k, max_k)

    x2 = np.sum(x * x, axis=1, keepdims=True)  # [N, 1]
    out_idx = np.empty((n, k_eff), dtype=np.int64)
    out_dist = np.empty((n, k_eff), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        q = x[start:end]  # [B, D]
        q2 = np.sum(q * q, axis=1, keepdims=True)  # [B, 1]
        # dist^2 = ||q||^2 + ||x||^2 - 2 qx^T
        dist2 = q2 + x2.T - 2.0 * (q @ x.T)
        np.maximum(dist2, 0.0, out=dist2)

        if exclude_self:
            rows = np.arange(start, end, dtype=np.int64)
            dist2[np.arange(end - start), rows] = np.inf

        part = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
        part_dist = np.take_along_axis(dist2, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        idx_sorted = np.take_along_axis(part, order, axis=1)
        dist_sorted = np.take_along_axis(part_dist, order, axis=1)

        out_idx[start:end] = idx_sorted
        out_dist[start:end] = np.sqrt(dist_sorted).astype(np.float32)

    if k_eff < k:
        pad_idx = np.full((n, k - k_eff), -1, dtype=np.int64)
        pad_dist = np.full((n, k - k_eff), np.inf, dtype=np.float32)
        out_idx = np.concatenate([out_idx, pad_idx], axis=1)
        out_dist = np.concatenate([out_dist, pad_dist], axis=1)

    return out_idx, out_dist


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Row-wise min-max normalization to [0, 1].
    """
    x_min = np.min(x, axis=1, keepdims=True)
    x_max = np.max(x, axis=1, keepdims=True)
    denom = np.maximum(x_max - x_min, eps)
    return (x - x_min) / denom


def build_joint_ranked_candidates(
    wrist_knn_indices: np.ndarray,
    head_embeddings: np.ndarray,
    future_action_features: np.ndarray,
    alpha: float,
    beta: float,
    top_m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each anchor i, rank candidates j in wrist-KNN set by:
      score(i,j) = alpha * d_head(i,j) + beta * d_action(i,j)

    Returns:
      candidate_indices: [N, top_m]
      candidate_scores:  [N, top_m]
      head_distances:    [N, top_m]
      action_distances:  [N, top_m]
    """
    if wrist_knn_indices.ndim != 2:
        raise ValueError("wrist_knn_indices must be 2D [N, K]")
    if head_embeddings.ndim != 2:
        raise ValueError("head_embeddings must be 2D [N, Dh]")
    if future_action_features.ndim != 2:
        raise ValueError("future_action_features must be 2D [N, Da']")
    if top_m <= 0:
        raise ValueError(f"top_m must be positive, got {top_m}")
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("alpha/beta must be finite numbers")

    n, k = wrist_knn_indices.shape
    if head_embeddings.shape[0] != n or future_action_features.shape[0] != n:
        raise ValueError("N mismatch across knn/head/action inputs")

    top_m_eff = min(top_m, k)
    out_idx = np.full((n, top_m), -1, dtype=np.int64)
    out_score = np.full((n, top_m), -np.inf, dtype=np.float32)
    out_dh = np.full((n, top_m), np.inf, dtype=np.float32)
    out_da = np.full((n, top_m), np.inf, dtype=np.float32)

    head = head_embeddings.astype(np.float32, copy=False)
    act = future_action_features.astype(np.float32, copy=False)

    for i in range(n):
        cand = wrist_knn_indices[i]
        valid_mask = (cand >= 0) & (cand < n)
        if not np.any(valid_mask):
            continue
        c = cand[valid_mask].astype(np.int64)

        dh = np.linalg.norm(head[c] - head[i : i + 1], axis=1)
        da = np.linalg.norm(act[c] - act[i : i + 1], axis=1)

        dh_n = _normalize_rows(dh[None, :])[0]
        da_n = _normalize_rows(da[None, :])[0]
        score = alpha * dh_n + beta * da_n

        order = np.argsort(score)[::-1][:top_m_eff]
        sel = c[order]

        out_idx[i, :top_m_eff] = sel
        out_score[i, :top_m_eff] = score[order].astype(np.float32)
        out_dh[i, :top_m_eff] = dh[order].astype(np.float32)
        out_da[i, :top_m_eff] = da[order].astype(np.float32)

    return out_idx, out_score, out_dh, out_da
