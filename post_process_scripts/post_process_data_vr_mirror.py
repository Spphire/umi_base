import json
import os
import os.path as osp
import shutil
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import lz4.frame
import numpy as np
import zarr
from loguru import logger
from tqdm import tqdm

from diffusion_policy.common.data_models import ActionType
from diffusion_policy.common.image_utils import resize_image
from diffusion_policy.common.space_utils import (
    homo_matrix_to_pose_9d_batch,
    pose_3d_9d_to_homo_matrix_batch,
)
from diffusion_policy.real_world.post_process_utils import DataPostProcessingManagerVR


@dataclass
class EpisodeRecord:
    timestamp: np.ndarray
    left_robot_tcp_pose: np.ndarray
    left_robot_gripper_width: np.ndarray
    left_eye_tcp_pose: np.ndarray
    left_eye_img: np.ndarray
    left_wrist_img: Optional[np.ndarray] = None
    right_eye_tcp_pose: Optional[np.ndarray] = None
    right_eye_img: Optional[np.ndarray] = None

    @property
    def length(self) -> int:
        return int(self.timestamp.shape[0])


def extract_archives(input_dir: str, debug: bool = False, max_files: int = 5) -> None:
    data_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tar.gz')])
    logger.info(f"Found {len(data_files)} archive file(s) in {input_dir}")

    for file_idx, data_file in enumerate(data_files):
        if debug and file_idx >= max_files:
            logger.info(f"[Debug] Stopping archive extraction after {max_files} files")
            break

        abs_path = osp.abspath(osp.join(input_dir, data_file))
        logger.info(f"Extracting {abs_path}...")
        try:
            with tarfile.open(abs_path, 'r:gz') as tar:
                tar.extractall(path=input_dir)
        except tarfile.ReadError:
            logger.info(f"Trying lz4 decompression for {abs_path}...")
            with lz4.frame.open(abs_path, 'rb') as lz4_file:
                with tarfile.open(fileobj=lz4_file, mode='r|') as tar:
                    tar.extractall(path=input_dir)


def find_record_sessions(data_dir: str) -> Dict[str, Dict[str, str]]:
    dst_paths: List[str] = []
    for subfolder in sorted([f.path for f in os.scandir(data_dir) if f.is_dir()]):
        try:
            if any(name.endswith('.bson') for name in os.listdir(subfolder)):
                dst_paths.append(subfolder)
        except PermissionError:
            continue

    record_sessions: Dict[str, Dict[str, str]] = {}
    for dst_path in dst_paths:
        meta_path = osp.join(dst_path, 'metadata.json')
        if not osp.exists(meta_path):
            logger.warning(f"metadata.json not found in {dst_path}, skipping")
            continue

        try:
            metadata = json.load(open(meta_path, 'r'))
            uuid = metadata['uuid']
            session_uuid = metadata.get('parent_uuid', uuid)
            camera_position = metadata.get('camera_position') or 'head'
            record_sessions.setdefault(session_uuid, {})[camera_position] = dst_path
        except Exception as exc:
            logger.error(f"Failed to parse {meta_path}: {exc}")

    return record_sessions


def preprocess_image_stream(
    images: np.ndarray,
    use_dino: bool = False,
    resize_target: Optional[tuple] = None,
) -> np.ndarray:
    images = np.asarray(images)
    if not use_dino:
        return np.ascontiguousarray(images)

    processed = []
    for image in images:
        if resize_target is None:
            processed_image = resize_image(image, mode='crop')
        else:
            processed_image = resize_image(image, mode='resize', target_size=resize_target)
        processed.append(processed_image)

    return np.ascontiguousarray(np.asarray(processed))


def smooth_and_pad_gripper_width(gripper_width: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(gripper_width, dtype=np.float32).copy()
    if values.ndim == 1:
        values = values[:, None]

    for idx in range(1, len(values) - 2):
        if abs(float(values[idx] - values[idx - 1])) > 0.15:
            values[idx] = (values[idx - 1] + values[idx + 2]) / 2.0

    while len(values) < target_len:
        values = np.concatenate([values, values[-1:]], axis=0)

    return values.astype(np.float32, copy=False)


def validate_episode_lengths(record: EpisodeRecord, session_id: str) -> bool:
    target_len = record.length
    arrays = {
        'left_robot_tcp_pose': record.left_robot_tcp_pose,
        'left_robot_gripper_width': record.left_robot_gripper_width,
        'left_eye_tcp_pose': record.left_eye_tcp_pose,
        'left_eye_img': record.left_eye_img,
    }
    if record.left_wrist_img is not None:
        arrays['left_wrist_img'] = record.left_wrist_img
    if record.right_eye_tcp_pose is not None:
        arrays['right_eye_tcp_pose'] = record.right_eye_tcp_pose
    if record.right_eye_img is not None:
        arrays['right_eye_img'] = record.right_eye_img

    bad = {name: value.shape[0] for name, value in arrays.items() if value.shape[0] != target_len}
    if bad:
        logger.warning(f"Session {session_id} skipped due to length mismatch: {bad}, expected={target_len}")
        return False
    return True


def build_episode_from_obs_dict(
    obs_dict: Dict[str, np.ndarray],
    session_id: str,
    use_dino: bool = False,
    resize_target: Optional[tuple] = None,
    gripper_width_bias: float = 0.0,
    gripper_width_scale: float = 1.0,
) -> Optional[EpisodeRecord]:
    required_keys = [
        'timestamp',
        'left_robot_tcp_pose',
        'left_robot_gripper_width',
        'left_eye_tcp_pose',
        'left_eye_img',
    ]
    missing = [key for key in required_keys if key not in obs_dict]
    if missing:
        logger.warning(f"Session {session_id} skipped, missing keys: {missing}")
        return None

    timestamp = np.asarray(obs_dict['timestamp'], dtype=np.float32)
    left_robot_tcp_pose = np.asarray(obs_dict['left_robot_tcp_pose'], dtype=np.float32)
    left_eye_tcp_pose = np.asarray(obs_dict['left_eye_tcp_pose'], dtype=np.float32)

    gripper_width = smooth_and_pad_gripper_width(obs_dict['left_robot_gripper_width'], len(timestamp))
    gripper_width = (gripper_width + gripper_width_bias) * gripper_width_scale
    gripper_width = gripper_width.astype(np.float32, copy=False)

    left_eye_img = preprocess_image_stream(obs_dict['left_eye_img'], use_dino=use_dino, resize_target=resize_target)

    left_wrist_img = None
    if 'left_wrist_img' in obs_dict:
        left_wrist_img = preprocess_image_stream(obs_dict['left_wrist_img'], use_dino=use_dino, resize_target=resize_target)

    right_eye_tcp_pose = None
    right_eye_img = None
    if 'right_eye_tcp_pose' in obs_dict and 'right_eye_img' in obs_dict:
        right_eye_tcp_pose = np.asarray(obs_dict['right_eye_tcp_pose'], dtype=np.float32)
        right_eye_img = preprocess_image_stream(obs_dict['right_eye_img'], use_dino=use_dino, resize_target=resize_target)

    record = EpisodeRecord(
        timestamp=timestamp,
        left_robot_tcp_pose=left_robot_tcp_pose,
        left_robot_gripper_width=gripper_width,
        left_eye_tcp_pose=left_eye_tcp_pose,
        left_eye_img=left_eye_img,
        left_wrist_img=left_wrist_img,
        right_eye_tcp_pose=right_eye_tcp_pose,
        right_eye_img=right_eye_img,
    )
    if not validate_episode_lengths(record, session_id):
        return None
    return record


def reflection_matrix_x() -> np.ndarray:
    reflection = np.eye(3, dtype=np.float32)
    reflection[0, 0] = -1.0
    return reflection


def mirror_pose_batch(pose_9d: np.ndarray) -> np.ndarray:
    mats = pose_3d_9d_to_homo_matrix_batch(np.asarray(pose_9d, dtype=np.float32))
    reflection = reflection_matrix_x()
    mirrored = mats.copy()
    mirrored[:, :3, :3] = reflection[None, :, :] @ mats[:, :3, :3] @ reflection[None, :, :]
    mirrored[:, :3, 3] = mats[:, :3, 3]
    mirrored[:, 0, 3] = -mats[:, 0, 3]
    return homo_matrix_to_pose_9d_batch(mirrored).astype(np.float32)


def flip_image_stream(images: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if images is None:
        return None
    return np.ascontiguousarray(np.flip(images, axis=2))


def mirror_episode(record: EpisodeRecord) -> EpisodeRecord:
    return EpisodeRecord(
        timestamp=record.timestamp.copy(),
        left_robot_tcp_pose=mirror_pose_batch(record.left_robot_tcp_pose),
        left_robot_gripper_width=record.left_robot_gripper_width.copy(),
        left_eye_tcp_pose=mirror_pose_batch(record.left_eye_tcp_pose),
        left_eye_img=flip_image_stream(record.left_eye_img),
        left_wrist_img=flip_image_stream(record.left_wrist_img),
        right_eye_tcp_pose=None if record.right_eye_tcp_pose is None else mirror_pose_batch(record.right_eye_tcp_pose),
        right_eye_img=flip_image_stream(record.right_eye_img),
    )


def create_absolute_actions(state_arrays: np.ndarray, episode_ends_arrays: np.ndarray) -> np.ndarray:
    next_state = state_arrays[1:, ...].copy()
    action_arrays = np.concatenate([next_state, next_state[-1:]], axis=0)

    for episode_end in episode_ends_arrays:
        if episode_end >= 2:
            action_arrays[episode_end - 1] = action_arrays[episode_end - 2]

    return action_arrays.astype(np.float32, copy=False)


def concatenate_optional_episode_field(episodes: List[EpisodeRecord], field: str) -> Optional[np.ndarray]:
    values = [getattr(episode, field) for episode in episodes]
    if all(value is not None for value in values):
        return np.concatenate(values, axis=0)
    if any(value is not None for value in values):
        logger.warning(f"Field {field} is missing in part of the episodes, omitting it from zarr output")
    return None


def concatenate_episodes(episodes: List[EpisodeRecord]) -> Dict[str, Optional[np.ndarray]]:
    if not episodes:
        raise ValueError('No episodes collected')

    episode_ends = []
    total_count = 0
    for episode in episodes:
        total_count += episode.length
        episode_ends.append(total_count)

    result = {
        'timestamp': np.concatenate([episode.timestamp for episode in episodes], axis=0).astype(np.float32, copy=False),
        'left_robot_tcp_pose': np.concatenate([episode.left_robot_tcp_pose for episode in episodes], axis=0).astype(np.float32, copy=False),
        'left_robot_gripper_width': np.concatenate([episode.left_robot_gripper_width for episode in episodes], axis=0).astype(np.float32, copy=False),
        'left_eye_tcp_pose': np.concatenate([episode.left_eye_tcp_pose for episode in episodes], axis=0).astype(np.float32, copy=False),
        'left_eye_img': np.concatenate([episode.left_eye_img for episode in episodes], axis=0),
        'left_wrist_img': concatenate_optional_episode_field(episodes, 'left_wrist_img'),
        'right_eye_tcp_pose': concatenate_optional_episode_field(episodes, 'right_eye_tcp_pose'),
        'right_eye_img': concatenate_optional_episode_field(episodes, 'right_eye_img'),
        'episode_ends': np.asarray(episode_ends, dtype=np.int64),
    }
    return result


def create_zarr_storage(
    save_data_path: str,
    timestamp_arrays: np.ndarray,
    left_robot_tcp_pose_arrays: np.ndarray,
    left_robot_gripper_width_arrays: np.ndarray,
    state_arrays: np.ndarray,
    action_arrays: np.ndarray,
    episode_ends_arrays: np.ndarray,
    left_wrist_img_arrays: Optional[np.ndarray] = None,
    left_eye_tcp_pose_arrays: Optional[np.ndarray] = None,
    left_eye_img_arrays: Optional[np.ndarray] = None,
    right_eye_tcp_pose_arrays: Optional[np.ndarray] = None,
    right_eye_img_arrays: Optional[np.ndarray] = None,
):
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    action_chunk_size = (10000, action_arrays.shape[1])
    if left_wrist_img_arrays is not None and len(left_wrist_img_arrays) > 0:
        image_chunk_size = (100, *left_wrist_img_arrays.shape[1:])
    elif left_eye_img_arrays is not None and len(left_eye_img_arrays) > 0:
        image_chunk_size = (100, *left_eye_img_arrays.shape[1:])
    else:
        image_chunk_size = (100, 480, 640, 3)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    zarr_data.create_dataset(
        'timestamp', data=timestamp_arrays, chunks=(10000,), dtype='float32', overwrite=True, compressor=compressor
    )
    zarr_data.create_dataset(
        'left_robot_tcp_pose', data=left_robot_tcp_pose_arrays, chunks=(10000, 9), dtype='float32', overwrite=True, compressor=compressor
    )
    zarr_data.create_dataset(
        'left_robot_gripper_width', data=left_robot_gripper_width_arrays, chunks=(10000, 1), dtype='float32', overwrite=True, compressor=compressor
    )
    zarr_data.create_dataset(
        'target', data=state_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor
    )
    zarr_data.create_dataset(
        'action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor
    )
    zarr_meta.create_dataset(
        'episode_ends', data=episode_ends_arrays, chunks=(10000,), dtype='int64', overwrite=True, compressor=compressor
    )

    if left_wrist_img_arrays is not None and len(left_wrist_img_arrays) > 0:
        zarr_data.create_dataset('left_wrist_img', data=left_wrist_img_arrays, chunks=image_chunk_size, dtype='uint8')
    if left_eye_tcp_pose_arrays is not None and len(left_eye_tcp_pose_arrays) > 0:
        zarr_data.create_dataset(
            'left_eye_tcp_pose', data=left_eye_tcp_pose_arrays, chunks=(10000, 9), dtype='float32', overwrite=True, compressor=compressor
        )
    if left_eye_img_arrays is not None and len(left_eye_img_arrays) > 0:
        zarr_data.create_dataset('left_eye_img', data=left_eye_img_arrays, chunks=image_chunk_size, dtype='uint8')
    if right_eye_tcp_pose_arrays is not None and len(right_eye_tcp_pose_arrays) > 0:
        zarr_data.create_dataset(
            'right_eye_tcp_pose', data=right_eye_tcp_pose_arrays, chunks=(10000, 9), dtype='float32', overwrite=True, compressor=compressor
        )
    if right_eye_img_arrays is not None and len(right_eye_img_arrays) > 0:
        zarr_data.create_dataset('right_eye_img', data=right_eye_img_arrays, chunks=image_chunk_size, dtype='uint8')

    return zarr_data, zarr_meta


def convert_data_to_zarr(
    input_dir: str,
    output_dir: str,
    use_absolute_action: bool = True,
    action_type: ActionType = ActionType.head_6DOF_left_arm_6DOF_gripper_width,
    debug: bool = False,
    overwrite: bool = True,
    use_dino: bool = False,
    resize_target: Optional[tuple] = None,
    gripper_width_bias: float = 0.0,
    gripper_width_scale: float = 1.0,
    episode_clip_head_seconds: float = 0.1,
    episode_clip_tail_seconds: float = 0.3,
    add_mirror_episode: bool = True,
    vr_alignment_mode: str = "legacy"
) -> str:
    if action_type != ActionType.head_6DOF_left_arm_6DOF_gripper_width:
        raise ValueError(
            'This script only supports ActionType.head_6DOF_left_arm_6DOF_gripper_width, '
            f'but got {action_type.name}'
        )

    save_data_path = osp.join(output_dir, 'replay_buffer.zarr')
    os.makedirs(output_dir, exist_ok=True)

    if osp.exists(save_data_path):
        if not overwrite:
            logger.info(f'Data already exists at {save_data_path}')
            return save_data_path
        logger.warning(f'Overwriting {save_data_path}')
        shutil.rmtree(save_data_path)

    extract_archives(input_dir, debug=debug)
    record_sessions = find_record_sessions(input_dir)
    if not record_sessions:
        logger.warning(f'No valid record sessions found in {input_dir}')
        return save_data_path

    data_processing_manager = DataPostProcessingManagerVR(use_6d_rotation=True, alignment_mode=vr_alignment_mode)
    episodes: List[EpisodeRecord] = []
    skipped_sessions = 0

    session_items = sorted(record_sessions.items(), key=lambda item: item[0])
    for session_uuid, session_paths in tqdm(session_items, desc='Processing sessions', dynamic_ncols=True):
        if debug and len(episodes) >= 10:
            logger.info('[Debug] Stopping session processing after collecting 10 episode variants')
            break

        obs_dict = data_processing_manager.extract_msg_to_obs_dict(
            session_paths,
            clip_head_seconds=episode_clip_head_seconds,
            clip_tail_seconds=episode_clip_tail_seconds,
            use_aruco_calibration=False,
        )
        if obs_dict is None:
            logger.warning(f'obs_dict is None for session {session_uuid}')
            skipped_sessions += 1
            continue

        episode = build_episode_from_obs_dict(
            obs_dict,
            session_uuid,
            use_dino=use_dino,
            resize_target=resize_target,
            gripper_width_bias=gripper_width_bias,
            gripper_width_scale=gripper_width_scale,
        )
        if episode is None:
            skipped_sessions += 1
            continue

        episodes.append(episode)
        if add_mirror_episode:
            episodes.append(mirror_episode(episode))

    if skipped_sessions > 0:
        logger.warning(f'{skipped_sessions} session(s) skipped during processing')
    if not episodes:
        raise RuntimeError('No episode was successfully converted')

    arrays = concatenate_episodes(episodes)
    state_arrays = np.concatenate(
        [
            arrays['left_robot_tcp_pose'],
            arrays['left_robot_gripper_width'],
            arrays['left_eye_tcp_pose'],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    if use_absolute_action:
        action_arrays = create_absolute_actions(state_arrays, arrays['episode_ends'])
    else:
        raise NotImplementedError('Only absolute actions are supported')

    zarr_data, zarr_meta = create_zarr_storage(
        save_data_path=save_data_path,
        timestamp_arrays=arrays['timestamp'],
        left_robot_tcp_pose_arrays=arrays['left_robot_tcp_pose'],
        left_robot_gripper_width_arrays=arrays['left_robot_gripper_width'],
        state_arrays=state_arrays,
        action_arrays=action_arrays,
        episode_ends_arrays=arrays['episode_ends'],
        left_wrist_img_arrays=arrays['left_wrist_img'],
        left_eye_tcp_pose_arrays=arrays['left_eye_tcp_pose'],
        left_eye_img_arrays=arrays['left_eye_img'],
        right_eye_tcp_pose_arrays=arrays['right_eye_tcp_pose'],
        right_eye_img_arrays=arrays['right_eye_img'],
    )

    original_episode_count = len(episodes) // 2 if add_mirror_episode else len(episodes)
    logger.info(f'Original episodes: {original_episode_count}')
    logger.info(f'Augmented episodes written: {len(arrays["episode_ends"])}')
    logger.info(f'Total frames: {len(arrays["timestamp"])}')
    logger.info('Zarr data structure:')
    logger.info(zarr_data.tree())
    logger.info(f'Total count after filtering: {action_arrays.shape[0]}')
    logger.info(f'Saved data at {save_data_path}')
    return save_data_path


if __name__ == '__main__':
    input_dir = '/mnt/data/shenyibo/workspace/umi_base/.cache/targz_q3_stack_cup'
    output_dir = '/mnt/data/shenyibo/workspace/q3_stack_cup_addon_mirror'
    debug = False
    use_absolute_action = True
    action_type = ActionType.head_6DOF_left_arm_6DOF_gripper_width
    overwrite = True
    use_dino = True
    resize_target = None
    gripper_width_bias = 0.0
    gripper_width_scale = 1.0
    add_mirror_episode = True
    zarr_path = convert_data_to_zarr(
        input_dir=input_dir,
        output_dir=output_dir,
        use_absolute_action=use_absolute_action,
        action_type=action_type,
        debug=debug,
        overwrite=overwrite,
        use_dino=use_dino,
        resize_target=resize_target,
        gripper_width_bias=gripper_width_bias,
        gripper_width_scale=gripper_width_scale,
        episode_clip_head_seconds=0.2,
        episode_clip_tail_seconds=0.2,
        add_mirror_episode=add_mirror_episode,
    )
    print(f'Data saved to {zarr_path}')
