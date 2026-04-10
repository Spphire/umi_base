from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from typing import Dict, Optional, Union

import lz4.frame
import requests
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from diffusion_policy.common.data_models import ActionType
from diffusion_policy.dataset.real_pick_and_place_image_head_relative_rotation_dataset import (
    RealPickAndPlaceImageHeadRelativeRotationDataset,
)


class CloudPickAndPlaceImageHeadRelativeRotationDataset(
    RealPickAndPlaceImageHeadRelativeRotationDataset
):
    """Cloud/local zarr wrapper for the relative-rotation image-head dataset.

    When ``local_files_only`` is provided, this class avoids importing the
    cloud-side post-processing stack, so local training with an existing zarr
    does not depend on heavy packages such as ``open3d``.
    """

    def __init__(
        self,
        local_files_only: Optional[str] = None,
        datacloud_endpoint: str = "http://127.0.0.1:8083",
        identifier: str = "Pick and place an empty cup",
        query_filter: Union[str, DictConfig, dict] = {},
        use_data_filtering=False,
        use_absolute_action=True,
        action_type: str = "left_arm_6DOF_gripper_width",
        temporal_downsample_ratio=0,
        temporal_upsample_ratio=0,
        use_dino=False,
        debug=False,
        episode_clip_head_seconds: float = 0.0,
        episode_clip_tail_seconds: float = 0.0,
        **kwargs,
    ):
        self.datacloud_endpoint = datacloud_endpoint
        self.identifier = identifier
        if isinstance(query_filter, str):
            self.query_filter = json.loads(query_filter)
        elif isinstance(query_filter, DictConfig):
            self.query_filter = OmegaConf.to_container(query_filter, resolve=True)
        elif isinstance(query_filter, dict):
            self.query_filter = query_filter
        else:
            raise ValueError("query_filter should be a dict or a JSON string.")

        self.use_data_filtering = use_data_filtering
        self.use_absolute_action = use_absolute_action
        self.action_type = ActionType[action_type]
        self.temporal_downsample_ratio = temporal_downsample_ratio
        self.temporal_upsample_ratio = temporal_upsample_ratio
        self.use_dino = use_dino
        self.episode_clip_head_seconds = episode_clip_head_seconds
        self.episode_clip_tail_seconds = episode_clip_tail_seconds

        self.config_hash = self._generate_config_hash()
        self.cache_dir = f".cache/cloud_pick_and_place_image_head_dataset/{self.config_hash}"

        if local_files_only is None:
            metadata = self._prepare_cloud_cache()
        else:
            metadata = {"zarr_path": local_files_only}

        zarr_path = metadata.get("zarr_path")
        assert zarr_path is not None, "Zarr path should not be None after cache validation."
        logger.info(f"Loading dataset from zarr path: {zarr_path}")

        if debug:
            self.zarr_path = zarr_path
        else:
            super().__init__(
                dataset_path=zarr_path,
                **kwargs,
            )

    def _generate_config_hash(self) -> str:
        config_dict = {
            "datacloud_endpoint": self.datacloud_endpoint,
            "identifier": self.identifier,
            "query_filter": self.query_filter,
            "use_data_filtering": self.use_data_filtering,
            "use_absolute_action": self.use_absolute_action,
            "action_type": str(self.action_type),
            "temporal_downsample_ratio": self.temporal_downsample_ratio,
            "temporal_upsample_ratio": self.temporal_upsample_ratio,
            "use_dino": self.use_dino,
            "episode_clip_head_seconds": self.episode_clip_head_seconds,
            "episode_clip_tail_seconds": self.episode_clip_tail_seconds,
        }
        config_str = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def _prepare_cloud_cache(self) -> Dict:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        cache_metadata_path = os.path.join(self.cache_dir, "metadata.json")
        if os.path.exists(cache_metadata_path):
            with open(cache_metadata_path, "r", encoding="utf-8") as file_obj:
                metadata = json.load(file_obj)
        else:
            metadata = {}

        list_recordings_request = {
            "identifier": self.identifier,
            "query_filter": self.query_filter,
            "limit": 10000,
            "skip": 0,
        }
        response = requests.post(
            f"{self.datacloud_endpoint}/v1/logs",
            json=list_recordings_request,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        records = response.json().get("data", [])
        assert len(records) > 0, "No records found for the given identifier."

        cloud_uuid_list = [record["uuid"] for record in records]
        logger.info(
            f"Found {len(cloud_uuid_list)} records in the cloud for identifier "
            f"'{self.identifier}' with query filter: {self.query_filter}."
        )
        cached_uuid_list = metadata.get("cached_uuid_list", [])
        if set(cloud_uuid_list) == set(cached_uuid_list):
            logger.info("Cache hit for cloud dataset.")
            return metadata

        logger.info("Cache miss for cloud dataset. Rebuilding zarr dataset.")
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, "downloaded_records.tar.lz4")
            data_request = {
                "identifier": self.identifier,
                "uuids": cloud_uuid_list,
            }
            response = requests.post(
                f"{self.datacloud_endpoint}/v1/download_records",
                json=data_request,
                stream=True,
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc="下载进度",
            )
            with open(archive_path, "wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file_obj.write(chunk)
                    progress_bar.update(len(chunk))
            progress_bar.close()

            extract_dir = os.path.join(temp_dir, "downloaded_records")
            os.makedirs(extract_dir, exist_ok=True)
            with lz4.frame.open(archive_path, "rb") as lz4_file:
                with tarfile.open(fileobj=lz4_file, mode="r|") as tar:
                    tar.extractall(path=extract_dir)

            from post_process_scripts.post_process_data_vr import convert_data_to_zarr

            zarr_path = convert_data_to_zarr(
                input_dir=extract_dir,
                output_dir=self.cache_dir,
                temporal_downsample_ratio=self.temporal_downsample_ratio,
                use_absolute_action=self.use_absolute_action,
                action_type=self.action_type,
                use_dino=self.use_dino,
                episode_clip_head_seconds=self.episode_clip_head_seconds,
                episode_clip_tail_seconds=self.episode_clip_tail_seconds,
            )

        metadata["cached_uuid_list"] = cloud_uuid_list
        metadata["zarr_path"] = zarr_path
        with open(cache_metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=4)
        logger.info(f"Zarr dataset built successfully at {zarr_path}")
        return metadata
