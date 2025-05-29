import pickle
import os
from loguru import logger
import zarr
import cv2
import numpy as np
import os.path as osp
import py_cli_interaction
import matplotlib.pyplot as plt
from hydra import initialize, compose
from omegaconf import DictConfig
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm
import tarfile

from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from diffusion_policy.common.visualization_utils import visualize_rgb_image
from diffusion_policy.real_world.post_process_utils import DataPostProcessingManagerPi0
from diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix

DEBUG = False
USE_DATA_FILTERING = False
USE_ABSOLUTE_ACTION = True
ACTION_DIM = 10  # (4 + 15)
TEMPORAL_DOWNSAMPLE_RATIO = 3
TEMPORAL_UPSAMPLE_RATIO = 0
# TEMPORAL_DOWNSAMPLE_RATIO = 0
# TEMPORAL_UPSAMPLE_RATIO = 5./3
# TEMPORAL_DOWNSAMPLE_RATIO = 0
# TEMPORAL_UPSAMPLE_RATIO = 0
SENSOR_MODE = 'single_arm_one_realsense'

def sixd_to_rotation_matrix(sixd):
    """
    将6D旋转表示转换为3x3旋转矩阵。
    6D表示由前两个列向量组成，第三列通过叉积计算获得正交基。
    """
    a1 = sixd[:, :3]
    a2 = sixd[:, 3:6]
    # 正交化
    a1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2 = a2 - np.sum(a1 * a2, axis=1, keepdims=True) * a1
    a2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
    a3 = np.cross(a1, a2)
    rotation_matrices = np.stack((a1, a2, a3), axis=2)  # shape: (N, 3, 3)
    return rotation_matrices

def rotation_matrix_to_sixd(rotation_matrices):
    """
    将3x3旋转矩阵转换回6D表示。
    返回前两列作为6D表示。
    """
    a1 = rotation_matrices[:, :, 0]
    a2 = rotation_matrices[:, :, 1]
    sixd = np.concatenate((a1, a2), axis=1)  # shape: (N, 6)
    return sixd

def interpolate_tcp_pose(tcp_pose, original_fps, target_fps):
    assert tcp_pose.shape[1] == 9, "tcp_pose should have 9 columns"
    num_original = tcp_pose.shape[0]
    duration = num_original / original_fps
    num_target = int(duration * target_fps)    
    original_times = np.linspace(0, duration, num=num_original, endpoint=False)
    target_times = np.linspace(0, duration, num=num_target, endpoint=False)
    target_times = np.clip(target_times, min(original_times), max(original_times))    
    interpolated_pose = np.zeros((num_target, tcp_pose.shape[1]))    # 分离tcp_pose的各个部分
    position = tcp_pose[:, :3]          # xyz
    sixd_rot = tcp_pose[:, 3:9]         # 6D rotation
    interp_func = interp1d(original_times, position, axis=0, kind='linear', fill_value="extrapolate")
    interpolated_pose[:, :3] = interp_func(target_times)

    # print("Interpolating rotation using SLERP...")
    # 转换6D到旋转矩阵
    rotation_matrices = sixd_to_rotation_matrix(sixd_rot)  # shape: (N, 3, 3)
    # 转换旋转矩阵到四元数
    rotations = R.from_matrix(rotation_matrices)
    # 定义slerp插值
    slerp = Slerp(original_times, rotations)
    # 插值
    interpolated_rotations = slerp(target_times)
    # 获取旋转矩阵
    interpolated_rotation_matrices = interpolated_rotations.as_matrix()
    # 转换旋转矩阵回6D
    interpolated_sixd = rotation_matrix_to_sixd(interpolated_rotation_matrices)
    # 插入到interpolated_pose
    interpolated_pose[:, 3:9] = interpolated_sixd    
    return interpolated_pose

def interpolate_common(common, original_fps, target_fps):
    num_original = common.shape[0]
    duration = num_original / original_fps
    num_target = int(duration * target_fps)    
    original_times = np.linspace(0, duration, num=num_original, endpoint=False)
    target_times = np.linspace(0, duration, num=num_target, endpoint=False)    
    interp_func = interp1d(original_times, common, axis=0, kind='linear', fill_value="extrapolate")
    interpolated_common = interp_func(target_times)
    return interpolated_common

def interpolate_images(images, original_fps, target_fps):
    num_original = images.shape[0]
    duration = num_original / original_fps
    num_target = int(duration * target_fps)    
    original_times = np.linspace(0, duration, num=num_original, endpoint=False)
    target_times = np.linspace(0, duration, num=num_target, endpoint=False)    
    interpolated_images = []    
    for i in range(num_target):
        t = target_times[i]
        # 找到最接近的两个原始帧
        if t >= original_times[-1]:
            frame = images[-1]
        else:
            idx = np.searchsorted(original_times, t) - 1
            idx = np.clip(idx, 0, num_original - 2)
            t1, t2 = original_times[idx], original_times[idx + 1]
            frame1, frame2 = images[idx], images[idx + 1]
            alpha = (t - t1) / (t2 - t1)
            # 混合两个帧
            frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        interpolated_images.append(frame) 

    return interpolated_images

if __name__ == '__main__':
    tag = 'real_pick_and_place_iphone'
    # tag = 'test'
    # we use the tag to determine if we want to use data filtering

    data_dir = f'data/{tag}'
    if USE_DATA_FILTERING:
        save_data_dir = f'data/{tag}_downsample{TEMPORAL_DOWNSAMPLE_RATIO}_filtered_zarr'
    else:
        save_data_dir = f'data/{tag}{"_debug" if DEBUG else ""}_zarr'
    save_data_path = osp.join(osp.join(osp.abspath(os.getcwd()), save_data_dir, f'replay_buffer.zarr'))
    os.makedirs(save_data_dir, exist_ok=True)
    if os.path.exists(save_data_path):
        logger.info('Data already exists at {}'.format(save_data_path))
        # use py_cli_interaction to ask user if they want to overwrite the data
        if py_cli_interaction.parse_cli_bool('Do you want to overwrite the data?', default_value=True):
            logger.warning('Overwriting {}'.format(save_data_path))
            os.system('rm -rf {}'.format(save_data_path))

    # create data processing manager
    data_processing_manager = DataPostProcessingManagerPi0(use_6d_rotation=True)

    # sensor data arrays
    timestamp_arrays = []
    left_wrist_img_arrays = []
    # robot state arrays
    left_robot_tcp_pose_arrays = []
    left_robot_gripper_width_arrays = []

    episode_ends_arrays = []
    total_count = 0
    # find all the files in the data directory
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tar.gz')])
    for seq_idx, data_file in enumerate(data_files):
        if DEBUG and seq_idx <= 25:
            continue
        data_path = osp.join(data_dir, data_file)
        abs_path = os.path.abspath(data_path)
        dst_path = abs_path.split('.tar.gz')[0]
        if not os.path.exists(dst_path):
            # continue
            os.makedirs(dst_path)
            logger.info(f"Extracting {abs_path}...")
            with tarfile.open(abs_path, 'r:gz') as tar:
                tar.extractall(path=dst_path)

        obs_dict = data_processing_manager.extract_msg_to_obs_dict(dst_path)
        if obs_dict is None:
            logger.warning(f"obs_dict is None for {dst_path}")
            continue
        timestamp_arrays.append(obs_dict['timestamp'])
        left_wrist_img_arrays.append(obs_dict['left_wrist_img'])
        left_robot_tcp_pose_arrays.append(obs_dict['left_robot_tcp_pose'])
        left_robot_gripper_width_arrays.append(obs_dict['left_robot_gripper_width'])
        total_count += len(obs_dict['timestamp'])
        episode_ends_arrays.append(total_count)
        while len(left_robot_gripper_width_arrays[-1]) < len(left_robot_tcp_pose_arrays[-1]):
            left_robot_gripper_width_arrays[-1] = np.concatenate([left_robot_gripper_width_arrays[-1], left_robot_gripper_width_arrays[-1][-1][np.newaxis, :]])

    # Convert lists to arrays
    left_wrist_img_arrays = np.vstack(left_wrist_img_arrays)
    episode_ends_arrays = np.array(episode_ends_arrays)
    timestamp_arrays = np.vstack(timestamp_arrays)
    left_robot_tcp_pose_arrays = np.vstack(left_robot_tcp_pose_arrays)
    left_robot_gripper_width_arrays = np.vstack(left_robot_gripper_width_arrays)

    if TEMPORAL_DOWNSAMPLE_RATIO > 1:
        # Calculate indices to keep after downsampling
        keep_indices = []
        current_episode_start = 0

        # Process each episode separately
        for episode_end in episode_ends_arrays:
            # Get indices for current episode
            episode_indices = np.arange(current_episode_start, episode_end)

            # Calculate downsampled indices for this episode
            # Keep first and last frame of each episode, downsample middle frames
            if len(episode_indices) > 2:
                middle_indices = episode_indices[1:-1]
                downsampled_middle_indices = middle_indices[::TEMPORAL_DOWNSAMPLE_RATIO]
                episode_keep_indices = np.concatenate([[episode_indices[0]],
                                                       downsampled_middle_indices,
                                                       [episode_indices[-1]]])
            else:
                # If episode is too short, keep all frames
                episode_keep_indices = episode_indices

            keep_indices.extend(episode_keep_indices)
            current_episode_start = episode_end

        keep_indices = np.array(keep_indices)

        # Downsample all arrays
        timestamp_arrays = timestamp_arrays[keep_indices]
        left_wrist_img_arrays = left_wrist_img_arrays[keep_indices]


        left_robot_tcp_pose_arrays = left_robot_tcp_pose_arrays[keep_indices]
        left_robot_gripper_width_arrays = left_robot_gripper_width_arrays[keep_indices]

        # Recalculate episode_ends
        new_episode_ends = []
        count = 0
        current_episode_start = 0
        for episode_end in episode_ends_arrays:
            episode_indices = np.arange(current_episode_start, episode_end)
            if len(episode_indices) > 2:
                middle_indices = episode_indices[1:-1]
                downsampled_middle_indices = middle_indices[::TEMPORAL_DOWNSAMPLE_RATIO]
                count += len(downsampled_middle_indices) + 2  # +2 for first and last frame
            else:
                count += len(episode_indices)
            new_episode_ends.append(count)
            current_episode_start = episode_end

        episode_ends_arrays = np.array(new_episode_ends)

    if ACTION_DIM == 4: # (left_tcp_x, left_tcp_y, left_tcp_z, left_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays[:, :3], left_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 10: # (left_tcp_x, left_tcp_y, left_tcp_z, left_6d_rotation, left_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays, left_robot_gripper_width_arrays], axis=-1)
    else:
        # TODO: support left_gripper1_marker_offset_emb_arrays
        # TODO: support right_gripper1_marker_offset_emb_arrays
        # TODO: support right_gripper2_marker_offset_emb_arrays
        raise NotImplementedError
    if USE_ABSOLUTE_ACTION:
        # override action to absolute value
        # action is basically next state
        new_action_arrays = state_arrays[1:, ...].copy()
        action_arrays = np.concatenate([new_action_arrays, new_action_arrays[-1][np.newaxis, :]], axis=0)
        # fix the last action of each episode
        for i in range(0, len(episode_ends_arrays)):
            action_arrays[episode_ends_arrays[i] - 1] = action_arrays[episode_ends_arrays[i] - 2]
    else:
        raise NotImplementedError
        
    valid_mask = np.ones(len(action_arrays), dtype=bool)


    # create zarr file4
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    # Compute chunk sizes
    wrist_img_chunk_size = (100, left_wrist_img_arrays.shape[1], left_wrist_img_arrays.shape[2], left_wrist_img_arrays.shape[3])
    if len(action_arrays.shape) == 2:
        action_chunk_size = (10000, action_arrays.shape[1])
    else:
        raise NotImplementedError

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    zarr_data.create_dataset('timestamp', data=timestamp_arrays, chunks=(10000,), dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('left_robot_tcp_pose', data=left_robot_tcp_pose_arrays, chunks=(10000, 9), dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('left_robot_gripper_width', data=left_robot_gripper_width_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('target', data=state_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(10000,), dtype='int64', overwrite=True,
                             compressor=compressor)

    zarr_data.create_dataset('left_wrist_img', data=left_wrist_img_arrays, chunks=wrist_img_chunk_size, dtype='uint8')
    

    # print zarr data structure
    logger.info('Zarr data structure')
    logger.info(zarr_data.tree())
    logger.info(f'Total count after filtering: {action_arrays.shape[0]}')
    logger.info(f'Save data at {save_data_path}')
