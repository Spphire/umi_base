#!/usr/bin/env python3
"""
Script for building zarr datasets from cloud data for training purposes.
This script downloads data from a cloud endpoint, processes it, and saves it as a zarr file.
"""

import os
import sys
import tempfile
import shutil
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import requests
import zarr
import cv2
from tqdm import tqdm
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf

# Configure logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# [local import]
if True:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusion_policy.real_world.post_process_utils import DataPostProcessingManageriPhone
from diffusion_policy.common.image_utils import center_crop_and_resize_image
from diffusion_policy.common.space_utils import pose_3d_9d_to_homo_matrix_batch, homo_matrix_to_pose_9d_batch

# []
@dataclass
class CloudDatasetConfig:
    """Configuration for cloud dataset creation."""
    datacloud_endpoint: str
    identifier: str
    start_times: List[str]
    end_times: List[str]
    num: Optional[int]
    num_start: int
    comment: str
    output_dir: str
    overwrite: bool
    use_data_filtering: bool
    use_absolute_action: bool
    action_dim: int
    temporal_downsample_ratio: int
    temporal_upsample_ratio: int
    use_dino: bool
    image_shape: List[int]
    image_resize_mode: str
    gripper_width_bias: float
    gripper_width_scale: float
    tcp_transform: List[List[float]]
    debug: bool


def to_time_ranges(start_times: List[str], end_times: List[str]) -> List[Dict[str, str]]:
    """Convert start and end times to time range format for API requests."""
    return [{"start_time": start, "end_time": end} for start, end in zip(start_times, end_times)]


def get_cloud_records(cfg: CloudDatasetConfig) -> Tuple[List[str], List[Dict]]:
    """Get list of record UUIDs from cloud endpoint."""
    url = f"{cfg.datacloud_endpoint}/v1/logs/{cfg.identifier}"
    try:
        params = {}
        if cfg.start_times and cfg.end_times:
            if len(cfg.start_times) != len(cfg.end_times):
                raise ValueError("start_times and end_times must have the same length")
            params["time_ranges"] = json.dumps(to_time_ranges(cfg.start_times, cfg.end_times))

        logger.info(f"Fetching records from {url}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        records = response.json().get('data', [])
        if len(records) == 0:
            logger.warning(f"No records found for identifier '{cfg.identifier}'")
            return [], []

        uuid_list = [record['uuid'] for record in records]
        logger.info(f"Found {len(uuid_list)} records in the cloud")

        # Apply num and num_start filters
        if cfg.num is not None:
            start_idx = cfg.num_start
            end_idx = start_idx + cfg.num
            if end_idx > len(uuid_list):
                logger.warning(f"Requested {cfg.num} records starting from {cfg.num_start}, but only {len(uuid_list)} available")
                end_idx = len(uuid_list)
            uuid_list = uuid_list[start_idx:end_idx]
            records = records[start_idx:end_idx]
            logger.info(f"Filtered to {len(uuid_list)} records (num={cfg.num}, num_start={cfg.num_start})")

        return uuid_list, records
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while fetching records from cloud: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch records from cloud: {str(e)}")
        raise


def download_records(cfg: CloudDatasetConfig, uuids: List[str], temp_dir: str) -> str:
    """Download records from cloud and extract them."""
    filename = os.path.join(temp_dir, "downloaded_records.tar.gz")

    try:
        data_request = {
            "identifier": cfg.identifier,
            "uuids": uuids,
        }

        logger.info(f"Downloading {len(uuids)} records from {cfg.datacloud_endpoint}")
        response = requests.post(
            f"{cfg.datacloud_endpoint}/v1/download_records",
            json=data_request,
            stream=True,
            timeout=300  # 5 minute timeout for large downloads
        )
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Download with progress bar
        with open(filename, 'wb') as f:
            if total_size > 0:
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
            else:
                progress_bar = tqdm(unit='B', unit_scale=True, desc="Downloading")

            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    chunk_size = len(chunk)
                    downloaded_size += chunk_size
                    progress_bar.update(chunk_size)

        progress_bar.close()

        # Verify checksum if provided
        server_sha256sum = response.headers.get('X-File-SHA256')
        if server_sha256sum:
            sha256_hash = hashlib.sha256()
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            file_sha256sum = sha256_hash.hexdigest()
            if file_sha256sum != server_sha256sum:
                raise ValueError(f"SHA256 checksum mismatch: {file_sha256sum} != {server_sha256sum}")
            else:
                logger.info("SHA256 verification successful")

    except Exception as e:
        logger.error(f"Failed to download data: {str(e)}")
        raise

    # Extract the downloaded file
    extract_dir = os.path.join(temp_dir, "downloaded_records")
    try:
        logger.info(f"Extracting to {extract_dir}")
        shutil.unpack_archive(filename, extract_dir, 'gztar')
    except Exception as e:
        logger.error(f"Failed to extract downloaded data: {str(e)}")
        raise

    return extract_dir


def convert_data_to_zarr(
    input_dir: str,
    output_path: str,
    cfg: CloudDatasetConfig
) -> str:
    """Convert downloaded data to zarr format."""

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if output already exists
    if os.path.exists(output_path):
        if cfg.overwrite:
            logger.warning(f"Overwriting existing zarr at {output_path}")
            shutil.rmtree(output_path)
        else:
            logger.info(f"Zarr already exists at {output_path}")
            return output_path

    # Create data processing manager
    try:
        data_processing_manager = DataPostProcessingManageriPhone(
            image_resize_shape=tuple(cfg.image_shape[1:]),  # (height, width)
            use_6d_rotation=True,
            debug=cfg.debug
        )
    except Exception as e:
        logger.error(f"Failed to create data processing manager: {str(e)}")
        raise

    # Initialize data arrays
    timestamp_arrays = []
    left_wrist_img_arrays = []
    left_robot_tcp_pose_arrays = []
    left_robot_gripper_width_arrays = []
    episode_ends_arrays = []
    total_count = 0

    # Convert TCP transform to numpy array
    tcp_transform = np.array(cfg.tcp_transform, dtype=np.float32)

    # Process each directory containing .bson files
    dst_paths = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            # Check if directory contains .bson files
            if any(f.endswith('.bson') for f in os.listdir(item_path)):
                dst_paths.append(item_path)

    if not dst_paths:
        raise ValueError(f"No .bson files found in {input_dir}")

    logger.info(f"Processing {len(dst_paths)} data directories")

    for dst_path in tqdm(dst_paths, desc="Processing episodes"):
        try:
            # Extract observation data
            obs_dict = data_processing_manager.extract_msg_to_obs_dict(dst_path)
            if obs_dict is None:
                logger.warning(f"Failed to extract data from {dst_path}")
                continue
        except Exception as e:
            logger.warning(f"Error processing {dst_path}: {str(e)}")
            continue

        # Collect data
        timestamp_arrays.append(obs_dict['timestamp'])

        # Transform robot TCP pose if needed
        if np.array_equal(tcp_transform, np.eye(4)):
            left_robot_tcp_pose_arrays.append(obs_dict['left_robot_tcp_pose'])
        else:
            transformed_poses = []
            for pose in obs_dict['left_robot_tcp_pose']:
                pose_array = pose[np.newaxis, :]
                pose_homo_matrix = pose_3d_9d_to_homo_matrix_batch(pose_array)
                transformed_tcp_matrix = tcp_transform @ pose_homo_matrix
                transformed_9d_pose = homo_matrix_to_pose_9d_batch(transformed_tcp_matrix).squeeze()
                transformed_poses.append(transformed_9d_pose)
            left_robot_tcp_pose_arrays.append(np.array(transformed_poses))

        total_count += len(obs_dict['timestamp'])
        episode_ends_arrays.append(total_count)

        # Process gripper width data
        gripper_width = obs_dict['left_robot_gripper_width']
        # Apply bias and scale
        gripper_width = (gripper_width + cfg.gripper_width_bias) * cfg.gripper_width_scale
        left_robot_gripper_width_arrays.append(gripper_width)

        # Process images
        if cfg.use_dino:
            processed_images = []
            for img in obs_dict['left_wrist_img']:
                processed_img = center_crop_and_resize_image(img, tuple(cfg.image_shape[1:]))
                processed_images.append(processed_img)
            left_wrist_img_arrays.append(np.array(processed_images))
        else:
            left_wrist_img_arrays.append(np.array(obs_dict['left_wrist_img']))

    # Convert lists to arrays
    if not timestamp_arrays:
        raise ValueError("No valid data found to process")

    timestamp_arrays = np.vstack(timestamp_arrays)
    left_robot_tcp_pose_arrays = np.vstack(left_robot_tcp_pose_arrays)
    left_robot_gripper_width_arrays = np.vstack(left_robot_gripper_width_arrays)
    left_wrist_img_arrays = np.vstack(left_wrist_img_arrays)
    episode_ends_arrays = np.array(episode_ends_arrays)

    # Apply temporal downsampling if needed
    if cfg.temporal_downsample_ratio > 1:
        timestamp_arrays, left_wrist_img_arrays, left_robot_tcp_pose_arrays, \
        left_robot_gripper_width_arrays, episode_ends_arrays = downsample_temporal_data(
            cfg.temporal_downsample_ratio,
            timestamp_arrays,
            left_wrist_img_arrays,
            left_robot_tcp_pose_arrays,
            left_robot_gripper_width_arrays,
            episode_ends_arrays
        )

    # Build state arrays
    if cfg.action_dim == 4:
        state_arrays = np.concatenate([
            left_robot_tcp_pose_arrays[:, :3],
            left_robot_gripper_width_arrays
        ], axis=-1)
    elif cfg.action_dim == 10:
        state_arrays = np.concatenate([
            left_robot_tcp_pose_arrays,
            left_robot_gripper_width_arrays
        ], axis=-1)
    else:
        raise ValueError(f"Unsupported action_dim: {cfg.action_dim}")

    # Build action arrays
    if cfg.use_absolute_action:
        action_arrays = create_absolute_actions(state_arrays, episode_ends_arrays)
    else:
        raise NotImplementedError("Only absolute actions are supported")

    # Create zarr storage
    create_zarr_storage(
        output_path,
        timestamp_arrays,
        left_robot_tcp_pose_arrays,
        left_robot_gripper_width_arrays,
        state_arrays,
        action_arrays,
        episode_ends_arrays,
        left_wrist_img_arrays
    )

    logger.info(f"Zarr dataset created at {output_path}")
    logger.info(f"Total episodes: {len(episode_ends_arrays)}")
    logger.info(f"Total timesteps: {action_arrays.shape[0]}")

    return output_path


def downsample_temporal_data(
    downsample_ratio: int,
    timestamp_arrays: np.ndarray,
    left_wrist_img_arrays: np.ndarray,
    left_robot_tcp_pose_arrays: np.ndarray,
    left_robot_gripper_width_arrays: np.ndarray,
    episode_ends_arrays: np.ndarray
) -> tuple:
    """Apply temporal downsampling to data arrays."""
    keep_indices = []
    current_episode_start = 0

    for episode_end in episode_ends_arrays:
        episode_indices = np.arange(current_episode_start, episode_end)

        if len(episode_indices) > 2:
            middle_indices = episode_indices[1:-1]
            downsampled_middle_indices = middle_indices[::downsample_ratio]
            episode_keep_indices = np.concatenate([
                [episode_indices[0]],
                downsampled_middle_indices,
                [episode_indices[-1]]
            ])
        else:
            episode_keep_indices = episode_indices

        keep_indices.extend(episode_keep_indices)
        current_episode_start = episode_end

    keep_indices = np.array(keep_indices)

    # Downsample all arrays
    timestamp_arrays = timestamp_arrays[keep_indices]
    left_wrist_img_arrays = left_wrist_img_arrays[keep_indices]
    left_robot_tcp_pose_arrays = left_robot_tcp_pose_arrays[keep_indices]
    left_robot_gripper_width_arrays = left_robot_gripper_width_arrays[keep_indices]

    # Recalculate episode ends
    new_episode_ends = []
    count = 0
    current_episode_start = 0

    for episode_end in episode_ends_arrays:
        episode_indices = np.arange(current_episode_start, episode_end)
        if len(episode_indices) > 2:
            middle_indices = episode_indices[1:-1]
            downsampled_middle_indices = middle_indices[::downsample_ratio]
            count += len(downsampled_middle_indices) + 2
        else:
            count += len(episode_indices)
        new_episode_ends.append(count)
        current_episode_start = episode_end

    episode_ends_arrays = np.array(new_episode_ends)

    return (
        timestamp_arrays,
        left_wrist_img_arrays,
        left_robot_tcp_pose_arrays,
        left_robot_gripper_width_arrays,
        episode_ends_arrays
    )


def create_absolute_actions(
    state_arrays: np.ndarray,
    episode_ends_arrays: np.ndarray
) -> np.ndarray:
    """Create absolute action arrays from state arrays."""
    new_action_arrays = state_arrays[1:, ...].copy()
    action_arrays = np.concatenate([
        new_action_arrays,
        new_action_arrays[-1][np.newaxis, :]
    ], axis=0)

    for i in range(len(episode_ends_arrays)):
        action_arrays[episode_ends_arrays[i] - 1] = action_arrays[episode_ends_arrays[i] - 2]

    return action_arrays


def create_zarr_storage(
    save_data_path: str,
    timestamp_arrays: np.ndarray,
    left_robot_tcp_pose_arrays: np.ndarray,
    left_robot_gripper_width_arrays: np.ndarray,
    state_arrays: np.ndarray,
    action_arrays: np.ndarray,
    episode_ends_arrays: np.ndarray,
    left_wrist_img_arrays: np.ndarray
):
    """Create zarr storage for the dataset."""
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    # Calculate chunk sizes
    wrist_img_chunk_size = (100, *left_wrist_img_arrays.shape[1:])
    action_chunk_size = (10000, action_arrays.shape[1])

    # Create compressor
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    # Create datasets
    zarr_data.create_dataset('timestamp', data=timestamp_arrays,
                           chunks=(10000,), dtype='float32',
                           overwrite=True, compressor=compressor)

    zarr_data.create_dataset('left_robot_tcp_pose', data=left_robot_tcp_pose_arrays,
                           chunks=(10000, 9), dtype='float32',
                           overwrite=True, compressor=compressor)

    zarr_data.create_dataset('left_robot_gripper_width', data=left_robot_gripper_width_arrays,
                           chunks=(10000, 1), dtype='float32',
                           overwrite=True, compressor=compressor)

    zarr_data.create_dataset('target', data=state_arrays,
                           chunks=action_chunk_size, dtype='float32',
                           overwrite=True, compressor=compressor)

    zarr_data.create_dataset('action', data=action_arrays,
                           chunks=action_chunk_size, dtype='float32',
                           overwrite=True, compressor=compressor)

    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays,
                           chunks=(10000,), dtype='int64',
                           overwrite=True, compressor=compressor)

    zarr_data.create_dataset('left_wrist_img', data=left_wrist_img_arrays,
                           chunks=wrist_img_chunk_size, dtype='uint8')


def generate_output_filename(cfg: CloudDatasetConfig) -> str:
    """Generate output filename based on configuration."""
    date_str = datetime.now().strftime("%Y%m%d")

    # Create filename components
    components = [cfg.identifier]

    if cfg.num is not None:
        components.append(f"num{cfg.num}")

    components.append(f"ds{date_str}")

    if cfg.comment:
        components.append(cfg.comment)

    filename = "_".join(components) + ".zarr"
    return filename


@hydra.main(version_base=None, config_path="../diffusion_policy/config/dataset", config_name="train_dataset")
def main(cfg: DictConfig) -> None:
    """Main function for building zarr dataset from cloud data."""

    try:
        # Convert DictConfig to CloudDatasetConfig
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        dataset_cfg = CloudDatasetConfig(**cfg_dict)

        logger.info("Starting dataset creation process")
        logger.info(f"Identifier: {dataset_cfg.identifier}")
        logger.info(f"Time ranges: {len(dataset_cfg.start_times)}")
        logger.info(f"Output directory: {dataset_cfg.output_dir}")

        # Validate configuration
        if not dataset_cfg.identifier:
            raise ValueError("Identifier cannot be empty")

        if dataset_cfg.start_times and dataset_cfg.end_times:
            if len(dataset_cfg.start_times) != len(dataset_cfg.end_times):
                raise ValueError("start_times and end_times must have the same length")

        # Get list of records from cloud
        uuids, records = get_cloud_records(dataset_cfg)
        if not uuids:
            logger.error("No records found to process")
            return

        # Generate output filename
        output_filename = generate_output_filename(dataset_cfg)
        output_path = os.path.join(dataset_cfg.output_dir, output_filename)

        # Create output directory if it doesn't exist
        os.makedirs(dataset_cfg.output_dir, exist_ok=True)

        # Create temporary directory for downloading and processing
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Download records
                logger.info(f"Downloading {len(uuids)} records...")
                extract_dir = download_records(dataset_cfg, uuids, temp_dir)

                # Convert to zarr format
                logger.info("Converting data to zarr format...")
                zarr_path = convert_data_to_zarr(extract_dir, output_path, dataset_cfg)

                logger.info("🎉 Dataset creation completed successfully")
                logger.info(f"Zarr file saved at: {zarr_path}")
                logger.info(f"Total episodes processed: {len(records)}")

            except Exception as e:
                logger.error(f"Dataset creation failed during processing: {str(e)}")
                raise

    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        logger.error("Please check your configuration and network connection")
        sys.exit(1)


if __name__ == "__main__":
    main()
