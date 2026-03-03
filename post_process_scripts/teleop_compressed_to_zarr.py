#!/usr/bin/env python3
"""
Convert compressed teleop episodes (MP4 + BSON) to replay_buffer.zarr.

Input folder format (from compress_episodes.py):
  input_dir/
    episode_xxx/
      HikCameraDevice_0.mp4 as left_eye_img
      IPhoneCameraDevice_0.mp4 as left_wrist_img
      Quest3Teleoperator_0.bson as [N, [x,y,z,qx,qy,qz,qw,gw]]
      metadata.json

Output zarr format follows post_process_data_vr_mouse.py:
  data/
    timestamp
    left_robot_tcp_pose
    left_robot_gripper_width
    target
    action
    left_wrist_img
    left_eye_img
  meta/
    episode_ends
"""

import argparse
import base64
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import zarr
from loguru import logger

# Add parent directory to path to import space_utils
sys.path.append(osp.join(osp.dirname(__file__), '..'))
from diffusion_policy.common.space_utils import pose_7d_to_pose_6d, pose_6d_to_pose_9d
from diffusion_policy.common.image_utils import center_crop_and_resize_image


try:
    import bson
except ImportError:
    bson = None


def decode_bson_array(bson_path: str) -> Optional[np.ndarray]:
    if bson is None:
        raise ImportError("pymongo is required. Install with: pip install pymongo")

    with open(bson_path, "rb") as f:
        doc = bson.decode(f.read())

    data_b64 = doc.get("data_b64", None)
    shape = doc.get("shape", None)
    dtype = doc.get("dtype", None)
    if data_b64 is None or shape is None or dtype is None:
        logger.warning(f"Invalid bson schema: {bson_path}")
        return None

    raw = base64.b64decode(data_b64)
    arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
    return arr


def load_mp4_frames(mp4_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {mp4_path}")
        return None

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # keep BGR uint8 to match common cv2 pipeline
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None
    return np.asarray(frames, dtype=np.uint8)


def trim_to_length(arr: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if len(arr) < n:
        return None
    return arr[:n]


def create_absolute_actions(state_arrays: np.ndarray, episode_ends_arrays: np.ndarray) -> np.ndarray:
    new_action_arrays = state_arrays[1:, ...].copy()
    action_arrays = np.concatenate([new_action_arrays, new_action_arrays[-1][np.newaxis, :]], axis=0)
    for i in range(len(episode_ends_arrays)):
        e = int(episode_ends_arrays[i])
        if e >= 2:
            action_arrays[e - 1] = action_arrays[e - 2]
    return action_arrays


def create_zarr_storage(
    save_data_path: str,
    timestamp_arrays: np.ndarray,
    left_robot_tcp_pose_arrays: np.ndarray,
    left_robot_gripper_width_arrays: np.ndarray,
    state_arrays: np.ndarray,
    action_arrays: np.ndarray,
    episode_ends_arrays: np.ndarray,
    left_wrist_img_arrays: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    left_eye_img_arrays: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
):
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    action_chunk_size = (10000, action_arrays.shape[1])
    def _first_image_shape(arr_or_list):
        if arr_or_list is None:
            return None
        if isinstance(arr_or_list, np.ndarray):
            return arr_or_list.shape[1:] if len(arr_or_list) > 0 else None
        if len(arr_or_list) == 0:
            return None
        return arr_or_list[0].shape[1:]

    wrist_shape = _first_image_shape(left_wrist_img_arrays)
    eye_shape = _first_image_shape(left_eye_img_arrays)

    if wrist_shape is not None:
        wrist_img_chunk_size = (100, *wrist_shape)
    elif eye_shape is not None:
        wrist_img_chunk_size = (100, *eye_shape)
    else:
        wrist_img_chunk_size = (100, 480, 640, 3)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    zarr_data.create_dataset("timestamp", data=timestamp_arrays, chunks=(10000,), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("left_robot_tcp_pose", data=left_robot_tcp_pose_arrays, chunks=(10000, left_robot_tcp_pose_arrays.shape[1]), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("left_robot_gripper_width", data=left_robot_gripper_width_arrays, chunks=(10000, 1), dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("target", data=state_arrays, chunks=action_chunk_size, dtype="float32", overwrite=True, compressor=compressor)
    zarr_data.create_dataset("action", data=action_arrays, chunks=action_chunk_size, dtype="float32", overwrite=True, compressor=compressor)
    zarr_meta.create_dataset("episode_ends", data=episode_ends_arrays, chunks=(10000,), dtype="int64", overwrite=True, compressor=compressor)

    if left_wrist_img_arrays is not None:
        if isinstance(left_wrist_img_arrays, np.ndarray):
            if len(left_wrist_img_arrays) > 0:
                zarr_data.create_dataset("left_wrist_img", data=left_wrist_img_arrays, chunks=wrist_img_chunk_size, dtype="uint8", overwrite=True)
        elif len(left_wrist_img_arrays) > 0:
            total_wrist = int(sum(arr.shape[0] for arr in left_wrist_img_arrays))
            wrist_ds = zarr_data.create_dataset(
                "left_wrist_img",
                shape=(total_wrist, *left_wrist_img_arrays[0].shape[1:]),
                chunks=wrist_img_chunk_size,
                dtype="uint8",
                overwrite=True,
            )
            write_idx = 0
            for arr in left_wrist_img_arrays:
                n = arr.shape[0]
                wrist_ds[write_idx:write_idx + n] = arr
                write_idx += n

    if left_eye_img_arrays is not None:
        if isinstance(left_eye_img_arrays, np.ndarray):
            if len(left_eye_img_arrays) > 0:
                zarr_data.create_dataset("left_eye_img", data=left_eye_img_arrays, chunks=wrist_img_chunk_size, dtype="uint8", overwrite=True)
        elif len(left_eye_img_arrays) > 0:
            total_eye = int(sum(arr.shape[0] for arr in left_eye_img_arrays))
            eye_ds = zarr_data.create_dataset(
                "left_eye_img",
                shape=(total_eye, *left_eye_img_arrays[0].shape[1:]),
                chunks=wrist_img_chunk_size,
                dtype="uint8",
                overwrite=True,
            )
            write_idx = 0
            for arr in left_eye_img_arrays:
                n = arr.shape[0]
                eye_ds[write_idx:write_idx + n] = arr
                write_idx += n


def convert_episodes_to_zarr(input_dir: str, output_dir: str, overwrite: bool = True) -> str:
    input_dir = osp.abspath(input_dir)
    output_dir = osp.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    save_data_path = osp.join(output_dir, "replay_buffer.zarr")
    if osp.exists(save_data_path):
        if not overwrite:
            logger.info(f"Zarr already exists: {save_data_path}")
            return save_data_path
        import shutil
        logger.warning(f"Overwriting: {save_data_path}")
        shutil.rmtree(save_data_path)

    episode_dirs = sorted([str(p) for p in Path(input_dir).iterdir() if p.is_dir() and p.name.startswith("episode_")])
    logger.info(f"Found {len(episode_dirs)} episode folders")
    if len(episode_dirs) == 0:
        raise RuntimeError("No episode folders found")

    timestamp_arrays = []
    left_robot_tcp_pose_arrays = []
    left_robot_gripper_width_arrays = []
    left_wrist_img_arrays = []
    left_eye_img_arrays = []

    episode_ends_arrays = []
    total_count = 0

    for epi_dir in episode_dirs:
        # strict mode: exact filenames
        teleop_bson = osp.join(epi_dir, "Quest3Teleoperator_0.bson")
        left_eye_mp4 = osp.join(epi_dir, "HikCameraDevice_0.mp4")
        left_wrist_mp4 = osp.join(epi_dir, "IPhoneCameraDevice_0.mp4")

        if not osp.exists(teleop_bson):
            logger.warning(f"Skip {epi_dir}: missing Quest3Teleoperator_0.bson")
            continue
        if not osp.exists(left_eye_mp4):
            logger.warning(f"Skip {epi_dir}: missing HikCameraDevice_0.mp4")
            continue
        if not osp.exists(left_wrist_mp4):
            logger.warning(f"Skip {epi_dir}: missing IPhoneCameraDevice_0.mp4")
            continue

        # Quest3Teleoperator_0.bson source order: [x,y,z,qx,qy,qz,qw,gw]
        teleop_arr = decode_bson_array(teleop_bson)
        if teleop_arr is None:
            logger.warning(f"Skip {epi_dir}: missing Quest3Teleoperator bson")
            continue
        teleop_arr = np.asarray(teleop_arr)
        if teleop_arr.ndim != 2 or teleop_arr.shape[1] < 8:
            logger.warning(f"Skip {epi_dir}: unexpected teleop shape {teleop_arr.shape}, expect [N,>=8]")
            continue

        # Extract xyz and quaternion, then convert to 9D pose
        left_xyz = teleop_arr[:, 0:3]  # [N, 3]
        left_quat = teleop_arr[:, [6, 3, 4, 5]]  # [N, 4] reorder to [qw, qx, qy, qz]
        left_pose_7d = np.concatenate([left_xyz, left_quat], axis=-1)  # [N, 7]
        # Convert 7D -> 6D -> 9D using space_utils functions
        left_tcp = np.array([pose_6d_to_pose_9d(pose_7d_to_pose_6d(pose)) for pose in left_pose_7d])  # [N, 9]
        left_gripper = teleop_arr[:, 7:8]

        # strict mode: both image streams required
        left_wrist = load_mp4_frames(left_wrist_mp4)
        left_eye_img = load_mp4_frames(left_eye_mp4)
        if left_wrist is None or left_eye_img is None:
            logger.warning(f"Skip {epi_dir}: failed to decode required mp4 files")
            continue

        # normalize dims
        left_tcp = np.asarray(left_tcp)
        left_gripper = np.asarray(left_gripper)
        if left_gripper.ndim == 1:
            left_gripper = left_gripper[:, None]

        lengths = [len(left_tcp), len(left_gripper), len(left_wrist), len(left_eye_img)]

        n = int(min(lengths))
        if n < 2:
            logger.warning(f"Skip {epi_dir}: too short ({n})")
            continue

        left_tcp = trim_to_length(left_tcp, n)
        left_gripper = trim_to_length(left_gripper, n)
        left_wrist = trim_to_length(left_wrist, n)
        left_eye_img = trim_to_length(left_eye_img, n)

        # Match post_process_data_vr_mouse behavior:
        # - wrist image: center-crop + resize to 224x224
        # - head image: aspect-ratio resize + pad to 224x224
        left_wrist = np.array([
            center_crop_and_resize_image(img, target_size=(224, 224), crop=True)
            for img in left_wrist
        ], dtype=np.uint8)
        left_eye_img = np.array([
            center_crop_and_resize_image(img, target_size=(224, 224), crop=False)
            for img in left_eye_img
        ], dtype=np.uint8)

        # timestamp fallback: frame index
        timestamp = np.arange(n, dtype=np.float32)

        timestamp_arrays.append(timestamp)
        left_robot_tcp_pose_arrays.append(left_tcp.astype(np.float32))
        left_robot_gripper_width_arrays.append(left_gripper.astype(np.float32))

        left_wrist_img_arrays.append(left_wrist.astype(np.uint8))
        left_eye_img_arrays.append(left_eye_img.astype(np.uint8))

        total_count += n
        episode_ends_arrays.append(total_count)
        logger.info(f"Episode {Path(epi_dir).name}: kept {n} frames")

    if len(episode_ends_arrays) == 0:
        raise RuntimeError("No valid episodes after parsing")

    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)
    timestamp_arrays = np.concatenate(timestamp_arrays, axis=0)
    left_robot_tcp_pose_arrays = np.concatenate(left_robot_tcp_pose_arrays, axis=0)
    left_robot_gripper_width_arrays = np.concatenate(left_robot_gripper_width_arrays, axis=0)

    left_wrist_img_arrays = left_wrist_img_arrays if len(left_wrist_img_arrays) > 0 else None
    left_eye_img_arrays = left_eye_img_arrays if len(left_eye_img_arrays) > 0 else None

    state_arrays = np.concatenate([
        left_robot_tcp_pose_arrays,
        left_robot_gripper_width_arrays,
    ], axis=-1).astype(np.float32)

    action_arrays = create_absolute_actions(state_arrays, episode_ends_arrays).astype(np.float32)

    create_zarr_storage(
        save_data_path=save_data_path,
        timestamp_arrays=timestamp_arrays,
        left_robot_tcp_pose_arrays=left_robot_tcp_pose_arrays,
        left_robot_gripper_width_arrays=left_robot_gripper_width_arrays,
        state_arrays=state_arrays,
        action_arrays=action_arrays,
        episode_ends_arrays=episode_ends_arrays,
        left_wrist_img_arrays=left_wrist_img_arrays,
        left_eye_img_arrays=left_eye_img_arrays,
    )

    logger.info(f"Saved zarr: {save_data_path}")
    logger.info(f"Total episodes: {len(episode_ends_arrays)}")
    logger.info(f"Total frames: {len(timestamp_arrays)}")
    return save_data_path


def main():
    parser = argparse.ArgumentParser(description="Convert compressed teleop episodes to replay_buffer.zarr")
    parser.add_argument("--input-dir", "-i", default="/mnt/data/shenyibo/workspace/umi_base/.cache/pack_teleop_nofisheye_mouse", help="Folder containing episode_xxx subfolders")
    parser.add_argument("--output-dir", "-o", default=".cache/teleop_nofisheye_mouse", help="Output folder for replay_buffer.zarr")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing zarr")
    args = parser.parse_args()

    convert_episodes_to_zarr(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=(not args.no_overwrite),
    )


if __name__ == "__main__":
    main()
