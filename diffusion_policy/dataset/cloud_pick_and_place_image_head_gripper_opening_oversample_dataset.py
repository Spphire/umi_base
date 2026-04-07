from typing import Optional
import copy
import os

import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from scipy.signal import savgol_filter

from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.cloud_pick_and_place_image_head_dataset import (
    CloudPickAndPlaceImageHeadDataset,
)


TREND_CLOSING = 0
TREND_FLAT = 1
TREND_OPENING = 2

TREND_NAME_TO_LABEL = {
    "closing": TREND_CLOSING,
    "flat": TREND_FLAT,
    "opening": TREND_OPENING,
}
TREND_LABEL_TO_NAME = {value: key for key, value in TREND_NAME_TO_LABEL.items()}


def classify_trend(
    arr: np.ndarray,
    window: int = 5,
    poly: int = 2,
    diff_thresh: float = 1e-4,
) -> np.ndarray:
    win = min(max(window, 11), len(arr) // 2 * 2 + 1)
    if len(arr) < win:
        return np.full(len(arr), TREND_FLAT, dtype=np.int8)

    smooth = savgol_filter(arr, window_length=win, polyorder=poly)
    diff = np.gradient(smooth)
    trend = np.full(len(arr), TREND_FLAT, dtype=np.int8)
    trend[diff > diff_thresh] = TREND_OPENING
    trend[diff < -diff_thresh] = TREND_CLOSING
    return trend


def find_trend_windows(trend: np.ndarray, min_len: int = 5) -> list[tuple[int, int, int]]:
    windows = []
    if len(trend) == 0:
        return windows

    start = 0
    current = int(trend[0])
    for i in range(1, len(trend)):
        if int(trend[i]) != current:
            if i - start < min_len and windows:
                prev = windows.pop()
                start = prev[0]
                current = prev[2]
            windows.append((start, i, current))
            start = i
            current = int(trend[i])

    if len(trend) - start < min_len and windows:
        prev = windows.pop()
        start = prev[0]
        current = prev[2]
    windows.append((start, len(trend), current))
    return windows


def merge_small_trend_windows(
    epi_width: np.ndarray,
    windows: list[tuple[int, int, int]],
    small_movement_ratio: float = 0.5,
) -> list[tuple[int, int, int]]:
    if len(windows) == 0:
        return windows

    max_gripper = np.max(epi_width) if len(epi_width) > 0 else 1.0
    min_drop = max_gripper * small_movement_ratio

    merged_windows = []
    i = 0
    while i < len(windows):
        window = windows[i]
        if window[2] in (TREND_OPENING, TREND_CLOSING):
            val_range = float(np.abs(epi_width[window[1] - 1] - epi_width[window[0]]))
            if val_range < min_drop:
                prev_flat = len(merged_windows) > 0 and merged_windows[-1][2] == TREND_FLAT
                next_flat = i + 1 < len(windows) and windows[i + 1][2] == TREND_FLAT
                if prev_flat and next_flat:
                    prev = merged_windows.pop()
                    next_window = windows[i + 1]
                    merged_windows.append((prev[0], next_window[1], TREND_FLAT))
                    i += 2
                    continue
                if prev_flat:
                    prev = merged_windows.pop()
                    merged_windows.append((prev[0], window[1], TREND_FLAT))
                    i += 1
                    continue
                if next_flat:
                    next_window = windows[i + 1]
                    merged_windows.append((window[0], next_window[1], TREND_FLAT))
                    i += 2
                    continue
                merged_windows.append((window[0], window[1], TREND_FLAT))
                i += 1
                continue
        merged_windows.append(window)
        i += 1
    return merged_windows


def compute_episode_trend_labels(
    epi_width: np.ndarray,
    window: int = 5,
    poly: int = 2,
    diff_thresh: float = 1e-4,
    min_len: int = 5,
    small_movement_ratio: float = 0.5,
) -> np.ndarray:
    trend = classify_trend(
        epi_width,
        window=window,
        poly=poly,
        diff_thresh=diff_thresh,
    )
    windows = find_trend_windows(trend, min_len=min_len)
    windows = merge_small_trend_windows(
        epi_width,
        windows,
        small_movement_ratio=small_movement_ratio,
    )

    labels = np.full(len(epi_width), TREND_FLAT, dtype=np.int8)
    for start, end, label in windows:
        labels[start:end] = label
    return labels


def compute_trend_labels(
    gripper_width: np.ndarray,
    episode_ends: np.ndarray,
    window: int = 5,
    poly: int = 2,
    diff_thresh: float = 1e-4,
    min_len: int = 5,
    small_movement_ratio: float = 0.5,
) -> np.ndarray:
    labels = np.full(len(gripper_width), TREND_FLAT, dtype=np.int8)
    start = 0
    for end in episode_ends:
        epi_width = gripper_width[start:end]
        labels[start:end] = compute_episode_trend_labels(
            epi_width,
            window=window,
            poly=poly,
            diff_thresh=diff_thresh,
            min_len=min_len,
            small_movement_ratio=small_movement_ratio,
        )
        start = end
    return labels


class CloudPickAndPlaceImageHeadGripperOpeningOversampleDataset(
    CloudPickAndPlaceImageHeadDataset
):
    def __init__(
        self,
        enable_gripper_trend_oversample: bool = True,
        trend_cache_path: Optional[str] = None,
        opening_repeat: int = 3,
        trend_repeat: Optional[int] = None,
        oversample_trends=None,
        trend_window: int = 5,
        trend_poly: int = 2,
        trend_diff_thresh: float = 1e-4,
        trend_min_window_len: int = 5,
        trend_small_movement_ratio: float = 0.5,
        opening_match_mode: str = "any",
        **kwargs,
    ):
        self.enable_gripper_trend_oversample = enable_gripper_trend_oversample
        self.trend_cache_path = self._resolve_trend_cache_path(trend_cache_path)
        self.opening_repeat = opening_repeat
        self.trend_repeat = trend_repeat if trend_repeat is not None else opening_repeat
        if oversample_trends is None:
            oversample_trends = ["opening"]
        self.oversample_trends = tuple(oversample_trends)
        self.oversample_trend_labels = self._resolve_oversample_trend_labels(self.oversample_trends)
        self.trend_window = trend_window
        self.trend_poly = trend_poly
        self.trend_diff_thresh = trend_diff_thresh
        self.trend_min_window_len = trend_min_window_len
        self.trend_small_movement_ratio = trend_small_movement_ratio
        self.opening_match_mode = opening_match_mode

        super().__init__(**kwargs)

        if not hasattr(self, "replay_buffer"):
            return

        self.trend_labels = self._load_or_compute_trend_labels()
        if self.enable_gripper_trend_oversample:
            self._apply_trend_oversample()

    def _resolve_trend_cache_path(self, trend_cache_path: Optional[str]) -> Optional[str]:
        if trend_cache_path not in (None, ""):
            return trend_cache_path

        try:
            output_dir = HydraConfig.get().runtime.output_dir
            return os.path.join(output_dir, "gripper_trend_labels.npz")
        except Exception:
            logger.warning(
                "Hydra runtime output_dir is unavailable, gripper trend labels will not be cached to disk."
            )
            return None

    def _load_or_compute_trend_labels(self) -> np.ndarray:
        gripper_width = self.replay_buffer["left_robot_gripper_width"][:].reshape(-1)
        episode_ends = self.replay_buffer.episode_ends[:]

        if self.trend_cache_path:
            cached = self._try_load_trend_cache(
                trend_cache_path=self.trend_cache_path,
                n_steps=len(gripper_width),
                episode_ends=episode_ends,
            )
            if cached is not None:
                logger.info(f"Loaded gripper trend cache from {self.trend_cache_path}")
                return cached

        labels = compute_trend_labels(
            gripper_width=gripper_width,
            episode_ends=episode_ends,
            window=self.trend_window,
            poly=self.trend_poly,
            diff_thresh=self.trend_diff_thresh,
            min_len=self.trend_min_window_len,
            small_movement_ratio=self.trend_small_movement_ratio,
        )

        if self.trend_cache_path:
            self._save_trend_cache(
                trend_cache_path=self.trend_cache_path,
                labels=labels,
                episode_ends=episode_ends,
            )
            logger.info(f"Saved gripper trend cache to {self.trend_cache_path}")

        opening_ratio = float(np.mean(labels == TREND_OPENING)) if len(labels) > 0 else 0.0
        logger.info(
            "Computed gripper trend labels: "
            f"n_steps={len(labels)}, opening_ratio={opening_ratio:.4f}"
        )
        return labels

    def _resolve_oversample_trend_labels(self, oversample_trends) -> tuple[int, ...]:
        labels = []
        for trend_name in oversample_trends:
            trend_key = str(trend_name).lower()
            if trend_key not in TREND_NAME_TO_LABEL:
                raise ValueError(
                    f"Unsupported oversample trend '{trend_name}'. "
                    f"Supported values: {list(TREND_NAME_TO_LABEL.keys())}"
                )
            labels.append(TREND_NAME_TO_LABEL[trend_key])
        return tuple(labels)

    def _try_load_trend_cache(
        self,
        trend_cache_path: str,
        n_steps: int,
        episode_ends: np.ndarray,
    ) -> Optional[np.ndarray]:
        if not os.path.exists(trend_cache_path):
            return None

        try:
            cache = np.load(trend_cache_path, allow_pickle=False)
            labels = cache["trend_labels"].astype(np.int8)
            cached_episode_ends = cache["episode_ends"]
            if len(labels) != n_steps or not np.array_equal(cached_episode_ends, episode_ends):
                logger.warning(
                    "Ignoring stale gripper trend cache due to shape mismatch: "
                    f"path={trend_cache_path}"
                )
                return None

            cache_params = {
                "trend_window": int(cache["trend_window"]),
                "trend_poly": int(cache["trend_poly"]),
                "trend_diff_thresh": float(cache["trend_diff_thresh"]),
                "trend_min_window_len": int(cache["trend_min_window_len"]),
                "trend_small_movement_ratio": float(cache["trend_small_movement_ratio"]),
            }
            current_params = {
                "trend_window": self.trend_window,
                "trend_poly": self.trend_poly,
                "trend_diff_thresh": self.trend_diff_thresh,
                "trend_min_window_len": self.trend_min_window_len,
                "trend_small_movement_ratio": self.trend_small_movement_ratio,
            }
            if cache_params != current_params:
                logger.warning(
                    "Ignoring stale gripper trend cache due to parameter mismatch: "
                    f"path={trend_cache_path}"
                )
                return None
            return labels
        except Exception as e:
            logger.warning(f"Failed to load gripper trend cache {trend_cache_path}: {e}")
            return None

    def _save_trend_cache(
        self,
        trend_cache_path: str,
        labels: np.ndarray,
        episode_ends: np.ndarray,
    ) -> None:
        cache_dir = os.path.dirname(trend_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            trend_cache_path,
            trend_labels=labels.astype(np.int8),
            episode_ends=episode_ends.astype(np.int64),
            trend_window=np.array(self.trend_window, dtype=np.int64),
            trend_poly=np.array(self.trend_poly, dtype=np.int64),
            trend_diff_thresh=np.array(self.trend_diff_thresh, dtype=np.float64),
            trend_min_window_len=np.array(self.trend_min_window_len, dtype=np.int64),
            trend_small_movement_ratio=np.array(
                self.trend_small_movement_ratio,
                dtype=np.float64,
            ),
        )

    def _window_has_target_trend(self, index_row: np.ndarray) -> bool:
        if self.opening_match_mode != "any":
            raise ValueError(
                f"Unsupported opening_match_mode: {self.opening_match_mode}. "
                "Only 'any' is supported."
            )

        buffer_start_idx, buffer_end_idx, _, _ = index_row
        label_start_idx = min(buffer_start_idx + self.n_latency_steps, buffer_end_idx)
        if label_start_idx >= buffer_end_idx:
            label_start_idx = buffer_start_idx
        return bool(
            np.any(
                np.isin(
                    self.trend_labels[label_start_idx:buffer_end_idx],
                    self.oversample_trend_labels,
                )
            )
        )

    def _apply_trend_oversample(self) -> None:
        base_indices = self.sampler.indices
        if len(base_indices) == 0:
            return
        if self.trend_repeat <= 1:
            logger.info("Skipping gripper trend oversample because trend_repeat <= 1.")
            return

        target_mask = np.zeros(len(base_indices), dtype=bool)
        for i, index_row in enumerate(base_indices):
            target_mask[i] = self._window_has_target_trend(index_row)

        target_count = int(np.sum(target_mask))
        target_names = [TREND_LABEL_TO_NAME[label] for label in self.oversample_trend_labels]
        target_names_str = ",".join(target_names)
        if target_count == 0:
            logger.warning(
                "No matching trend windows found for gripper trend oversampling: "
                f"trends={target_names_str}"
            )
            return

        extra_indices = np.repeat(
            base_indices[target_mask],
            repeats=self.trend_repeat - 1,
            axis=0,
        )
        self.sampler.indices = np.concatenate([base_indices, extra_indices], axis=0)
        oversampled_target_ratio = (
            target_count * self.trend_repeat / len(self.sampler.indices)
        )
        logger.info(
            "Applied gripper trend oversample: "
            f"base_windows={len(base_indices)}, "
            f"target_windows={target_count}, "
            f"trends={target_names_str}, "
            f"trend_repeat={self.trend_repeat}, "
            f"total_windows={len(self.sampler.indices)}, "
            f"target_ratio_after={oversampled_target_ratio:.4f}"
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            key_first_k=self.key_first_k,
            dagger_sampling_ratio=self.dagger_sampling_ratio,
        )
        val_set.val_mask = ~self.val_mask
        val_set.enable_gripper_trend_oversample = False
        return val_set
