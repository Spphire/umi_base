from typing import Dict, Optional, Union
import numpy as np
import os
import tempfile
import shutil
import hashlib
import json
import requests
import tarfile
import lz4.frame
from diffusion_policy.dataset.real_pick_and_place_image_head_dataset import RealPickAndPlaceImageHeadDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.data_models import ActionType
from loguru import logger
from post_process_scripts.post_process_data_vr import convert_data_to_zarr
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

class CloudPickAndPlaceImageHeadDataset(RealPickAndPlaceImageHeadDataset):
    """
    Dataset that loads zarr data from cloud storage (e.g., S3, GCS, etc.)
    Inherits all functionality from RealPickAndPlaceImageDataset but overrides
    data loading to support cloud sources.
    """
    
    def __init__(self,
            # shape_meta: dict,
            # dataset_path: str,
            # horizon=1,
            # pad_before=0,
            # pad_after=0,
            # n_obs_steps=None,
            # n_latency_steps=0,
            # seed=42,
            # val_ratio=0.0,
            # max_train_episodes=None,
            # delta_action=False,
            # relative_action=False,
            local_files_only: Optional[str]=None,
            datacloud_endpoint: str="http://127.0.0.1:8083",
            identifier: str='Pick and place an empty cup',
            query_filter: Union[str, DictConfig, dict]={},
            use_data_filtering=False,
            use_absolute_action=True,
            action_type: str = 'left_arm_6DOF_gripper_width',
            temporal_downsample_ratio=0,
            temporal_upsample_ratio=0,
            use_dino=False,
            debug=False,
            episode_clip_head_seconds: float = 0.0,
            episode_clip_tail_seconds: float = 0.0,
            **kwargs
        ):
        """
        Initialize the dataset with cloud storage support. Besides the parameters from the parent class,
        the following parameters are specific to cloud storage:
          - datacloud_endpoint: The endpoint URL for the cloud storage service.
        """
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
        self.cache_dir = '.cache/cloud_pick_and_place_image_head_dataset/{}'.format(self.config_hash)

        # Step1-6: Prepare the cloud cache and validate metadata
        if local_files_only is None:
            metadata = self._prepare_cloud_cache()
        else:
            metadata = { 'zarr_path': local_files_only }
        
        # Step7: Load the zarr dataset
        zarr_path = metadata.get('zarr_path')
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
        """
        Generate a hash based on the dataset configuration to ensure unique identification.
        This is useful for caching and avoiding redundant downloads.
        """
        config_dict = {
            'datacloud_endpoint': self.datacloud_endpoint,
            'identifier': self.identifier,
            'query_filter': self.query_filter,
            'use_data_filtering': self.use_data_filtering,
            'use_absolute_action': self.use_absolute_action,
            'action_type': str(self.action_type),
            'temporal_downsample_ratio': self.temporal_downsample_ratio,
            'temporal_upsample_ratio': self.temporal_upsample_ratio,
            'use_dino': self.use_dino,
            'episode_clip_head_seconds': self.episode_clip_head_seconds,
            'episode_clip_tail_seconds': self.episode_clip_tail_seconds,
        }

        config_str = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
        hash_object = hashlib.sha256(config_str.encode('utf-8'))
        return hash_object.hexdigest()[:16]

    def _prepare_cloud_cache(self) -> Dict:
        # Step1: check cache directory and metadata
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_metadata_path = os.path.join(self.cache_dir, 'metadata.json')
        if os.path.exists(self.cache_metadata_path):
            with open(self.cache_metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Step2: check if the local cache is valid. If not, rebuild zarr cache.
        list_recordings_request = {
            "identifier": self.identifier,
            "query_filter": self.query_filter,
            "limit": 10000,
            "skip": 0,
        }
        url = f"{self.datacloud_endpoint}/v1/logs"
        try:
            response = requests.post(
                url,
                json=list_recordings_request,
                headers={"Content-Type": "application/json"}
            )
        except Exception as e:
            print(f"Request failed: {str(e)}")
            return None
        self.records = response.json().get('data', [])
        assert len(self.records) > 0, "No records found for the given identifier."

        cloud_uuid_list = [record['uuid'] for record in self.records]
        logger.info(f"Found {len(cloud_uuid_list)} records in the cloud for identifier '{self.identifier}' with query filter: {self.query_filter}.")
        cached_uuid_list = metadata.get('cached_uuid_list', [])
        if set(cloud_uuid_list) != set(cached_uuid_list):
            logger.info("Cache miss for cloud dataset. Rebuilding zarr dataset.")

            # Step3: Download the data from cloud and build the zarr dataset
            with tempfile.TemporaryDirectory() as temp_dir:
                filename = os.path.join(temp_dir, "downloaded_records.tar.lz4")
                try:
                    data_request = {
                        "identifier": self.identifier,
                        "uuids": cloud_uuid_list,
                    }
                    response = requests.post(
                        f"{self.datacloud_endpoint}/v1/download_records",
                        json=data_request,
                        stream=True
                    )

                    total_size = int(response.headers.get('content-length', 0))
            
                    if total_size > 0:
                        # 如果知道总大小，显示百分比进度条
                        progress_bar = tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            desc="下载进度"
                        )
                    else:
                        # 如果不知道总大小，显示下载速度和已下载大小
                        progress_bar = tqdm(
                            unit='B',
                            unit_scale=True,
                            desc="下载进度"
                        )

                    downloaded_size = 0
                    if response.status_code == 200:
                        with open(filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=1024*1024):
                                if chunk:
                                    f.write(chunk)
                                    chunk_size = len(chunk)
                                    downloaded_size += chunk_size
                                    progress_bar.update(chunk_size)

                        server_sha256sum = response.headers.get('X-File-SHA256')
                        if server_sha256sum:
                            sha256_hash = hashlib.sha256()
                            with open(filename, 'rb') as f:
                                for chunk in iter(lambda: f.read(4096), b""):
                                    sha256_hash.update(chunk)
                            file_sha256sum = sha256_hash.hexdigest()
                            if file_sha256sum != server_sha256sum:
                                logger.error(f"SHA256 checksum mismatch: {file_sha256sum} != {server_sha256sum}")
                                return None
                            else:
                                logger.info(f"Downloaded file SHA256: {server_sha256sum}, verification successful.")
                        else:
                            logger.warning("SHA256 checksum not provided in response headers.")
                    else:
                        print(f"Error response: {response.text}")
                except Exception as e:
                    logger.error(f"Failed to download data from cloud: {str(e)}")
                    return None
                
                # Step4: Extract the downloaded tar.lz4 file
                try:
                    extract_dir = os.path.join(temp_dir, "downloaded_records")
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    # Decompress lz4 and extract tar in one go
                    with lz4.frame.open(filename, 'rb') as lz4_file:
                        with tarfile.open(fileobj=lz4_file, mode='r|') as tar:
                            tar.extractall(path=extract_dir)
                    logger.info(f"Successfully extracted tar.lz4 file to {extract_dir}")
                except Exception as e:
                    logger.error(f"Failed to extract downloaded data: {str(e)}")
                    return None
                
                # Step5: Build the zarr dataset
                zarr_path = convert_data_to_zarr(
                    input_dir=os.path.join(temp_dir, "downloaded_records"),
                    output_dir=self.cache_dir,
                    temporal_downsample_ratio=self.temporal_downsample_ratio,
                    use_absolute_action=self.use_absolute_action,
                    action_type=self.action_type,
                    use_dino=self.use_dino,
                    episode_clip_head_seconds=self.episode_clip_head_seconds,
                    episode_clip_tail_seconds=self.episode_clip_tail_seconds
                )

                # Step6: Update metadata with the new cache
                metadata['cached_uuid_list'] = cloud_uuid_list
                metadata['zarr_path'] = zarr_path
                with open(self.cache_metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                logger.info(f"Zarr dataset built successfully at {zarr_path}")
                
        else:
            logger.info("Cache hit for cloud dataset.")

        return metadata