from typing import Dict, List
import numpy as np
import os
import tempfile
import shutil
import hashlib
import json
import requests
from diffusion_policy.dataset.real_pick_and_place_image_dataset import RealPickAndPlaceImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from loguru import logger
from post_process_scripts.post_process_data_iphone import convert_data_to_zarr
from tqdm import tqdm

class CloudPickAndPlaceImageDataset(RealPickAndPlaceImageDataset):
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
            datacloud_endpoint: str="http://127.0.0.1:8083",
            identifier: str='Pick and place an empty cup',
            use_data_filtering=False,
            use_absolute_action=True,
            action_dim=10,
            temporal_downsample_ratio=0,
            temporal_upsample_ratio=0,
            use_dino=False,
            debug=False,
            **kwargs
        ):
        """
        Initialize the dataset with cloud storage support. Besides the parameters from the parent class,
        the following parameters are specific to cloud storage:
          - datacloud_endpoint: The endpoint URL for the cloud storage service.
        """
        self.datacloud_endpoint = datacloud_endpoint
        self.identifier = identifier

        self.use_data_filtering = use_data_filtering
        self.use_absolute_action = use_absolute_action
        self.action_dim = action_dim
        self.temporal_downsample_ratio = temporal_downsample_ratio
        self.temporal_upsample_ratio = temporal_upsample_ratio
        self.use_dino = use_dino

        self.config_hash = self._generate_config_hash()
        self.cache_dir = '.cache/cloud_pick_and_place_image_dataset/{}'.format(self.config_hash)

        # Step1-6: Prepare the cloud cache and validate metadata
        metadata = self._prepare_cloud_cache()
        
        # Step7: Load the zarr dataset
        zarr_path = metadata.get('zarr_path', None).split('replay_buffer.zarr')[0]
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
            'use_data_filtering': self.use_data_filtering,
            'use_absolute_action': self.use_absolute_action,
            'action_dim': self.action_dim,
            'temporal_downsample_ratio': self.temporal_downsample_ratio,
            'temporal_upsample_ratio': self.temporal_upsample_ratio,
            'use_dino': self.use_dino,
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
        url = f"{self.datacloud_endpoint}/v1/logs/{self.identifier}"
        try:
            response = requests.get(url)
        except Exception as e:
            print(f"Request failed: {str(e)}")
            return None
        self.records = response.json().get('data', [])
        assert len(self.records) > 0, "No records found for the given identifier."

        cloud_uuid_list = [record['uuid'] for record in self.records]
        logger.info(f"Found {len(cloud_uuid_list)} records in the cloud for identifier '{self.identifier}'.")
        cached_uuid_list = metadata.get('cached_uuid_list', [])
        if set(cloud_uuid_list) != set(cached_uuid_list):
            logger.info("Cache miss for cloud dataset. Rebuilding zarr dataset.")

            # Step3: Download the data from cloud and build the zarr dataset
            with tempfile.TemporaryDirectory() as temp_dir:
                filename = os.path.join(temp_dir, "downloaded_records.tar.gz")
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
                
                # Step4: Extract the downloaded tar.gz file
                try:
                    shutil.unpack_archive(filename, os.path.join(temp_dir, "downloaded_records"), 'gztar')
                except Exception as e:
                    logger.error(f"Failed to extract downloaded data: {str(e)}")
                    return None
                
                # Step5: Build the zarr dataset
                zarr_path = convert_data_to_zarr(
                    input_dir=os.path.join(temp_dir, "downloaded_records"),
                    output_dir=self.cache_dir,
                    temporal_downsample_ratio=self.temporal_downsample_ratio,
                    use_absolute_action=self.use_absolute_action,
                    action_dim=self.action_dim,
                    use_dino=self.use_dino
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