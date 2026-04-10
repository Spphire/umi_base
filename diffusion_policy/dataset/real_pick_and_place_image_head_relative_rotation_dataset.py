from __future__ import annotations

import copy
from typing import Dict

import numpy as np

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
)
from diffusion_policy.dataset.real_pick_and_place_image_head_dataset import (
    RealPickAndPlaceImageHeadDataset,
)


class RealPickAndPlaceImageHeadRelativeRotationDataset(
    RealPickAndPlaceImageHeadDataset
):
    """Image-head dataset with virtual sin/cos low-dim features from zarr angles.

    The underlying zarr is expected to contain scalar angle fields named
    ``relative_yaw`` and ``pitch``. This dataset exposes them as two virtual
    low-dim observations:

    - ``relative_yaw_sincos`` -> [sin(yaw), cos(yaw)]
    - ``pitch_sincos`` -> [sin(pitch), cos(pitch)]
    """

    ANGLE_FEATURE_SOURCES = {
        "relative_yaw_sincos": "relative_yaw",
        "pitch_sincos": "pitch",
    }

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        angle_unit: str = "degrees",
        **kwargs,
    ):
        self.angle_unit = angle_unit.lower()
        if self.angle_unit not in {"degrees", "radians"}:
            raise ValueError(f"Unsupported angle_unit: {angle_unit}")
        original_shape_meta = copy.deepcopy(shape_meta)
        base_shape_meta = self._strip_virtual_lowdim_keys(original_shape_meta)

        super().__init__(
            shape_meta=base_shape_meta,
            dataset_path=dataset_path,
            **kwargs,
        )

        self.virtual_lowdim_keys = [
            key
            for key in original_shape_meta["obs"].keys()
            if key in self.ANGLE_FEATURE_SOURCES
        ]
        self._inject_virtual_lowdim_arrays()

        self.shape_meta = original_shape_meta
        self.lowdim_keys = [
            key
            for key, attr in original_shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "low_dim"
        ]
        if self.n_obs_steps is not None:
            for key in self.virtual_lowdim_keys:
                self.key_first_k[key] = self.n_obs_steps
        self.sampler.keys = (
            list(self.rgb_keys)
            + list(self.lowdim_keys)
            + list(getattr(self, "auxiliary_lowdim_keys", []))
            + ["action"]
        )

    @classmethod
    def _strip_virtual_lowdim_keys(cls, shape_meta: dict) -> dict:
        base_shape_meta = copy.deepcopy(shape_meta)
        obs_meta = base_shape_meta["obs"]
        for key in list(obs_meta.keys()):
            if key in cls.ANGLE_FEATURE_SOURCES:
                source_key = cls.ANGLE_FEATURE_SOURCES[key]
                if source_key not in obs_meta:
                    obs_meta[source_key] = {
                        "horizon": obs_meta[key]["horizon"],
                        "shape": [1],
                        "type": "low_dim",
                    }
                del obs_meta[key]
        return base_shape_meta

    def _angles_to_sincos(self, angles: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, dtype=np.float32)
        if angles.ndim == 2 and angles.shape[1] == 1:
            angles = angles[:, 0]
        if angles.ndim != 1:
            raise ValueError(f"Expected angle array of shape (T,) or (T,1), got {angles.shape}")
        if self.angle_unit == "degrees":
            angles = np.deg2rad(angles)
        return np.stack([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)

    def _inject_virtual_lowdim_arrays(self) -> None:
        for feature_key in self.virtual_lowdim_keys:
            source_key = self.ANGLE_FEATURE_SOURCES[feature_key]
            if source_key not in self.replay_buffer:
                raise KeyError(
                    f"Missing source angle field '{source_key}' required for '{feature_key}'."
                )
            self.replay_buffer.root["data"][feature_key] = self._angles_to_sincos(
                self.replay_buffer[source_key][:]
            )

    def get_normalizer(self, **kwargs):
        normalizer = super().get_normalizer(**kwargs)
        for key in self.virtual_lowdim_keys:
            stats = array_to_stats(self.replay_buffer[key][:])
            normalizer[key] = get_identity_normalizer_from_stat(stats)
        return normalizer
