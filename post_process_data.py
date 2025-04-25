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

from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from diffusion_policy.common.visualization_utils import visualize_rgb_image
from diffusion_policy.real_world.post_process_utils import DataPostProcessingManager
from diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix

DEBUG = False
USE_DATA_FILTERING = False
USE_ABSOLUTE_ACTION = True
ACTION_DIM = 10  # (4 + 15)
# TEMPORAL_DOWNSAMPLE_RATIO = 3
# TEMPORAL_UPSAMPLE_RATIO = 0
TEMPORAL_DOWNSAMPLE_RATIO = 0
TEMPORAL_UPSAMPLE_RATIO = 5./3
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
    tag = 'real_pick_and_place_pi0'
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

    # loading config for transforms
    with initialize(config_path='diffusion_policy/config', version_base="1.3"):
        # config is relative to a module
        cfg = compose(config_name="real_world_env")
    transforms = RealWorldTransforms(option=cfg.task.transforms)

    # create data processing manager
    data_processing_manager = DataPostProcessingManager(transforms=transforms,
                                                        mode=SENSOR_MODE,
                                                        use_6d_rotation=True,
                                                        )

    # sensor data arrays
    timestamp_arrays = []
    external_img_arrays = []
    left_wrist_img_arrays = []
    right_wrist_img_arrays = []
    # robot state arrays
    left_robot_tcp_pose_arrays = []
    left_robot_tcp_vel_arrays = []
    left_robot_tcp_wrench_arrays = []
    left_robot_gripper_width_arrays = []
    left_robot_gripper_force_arrays = []
    right_robot_tcp_pose_arrays = []
    right_robot_tcp_vel_arrays = []
    right_robot_tcp_wrench_arrays = []
    right_robot_gripper_width_arrays = []
    right_robot_gripper_force_arrays = []
    episode_fps = []

    episode_ends_arrays = []
    total_count = 0
    # find all the files in the data directory
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    for seq_idx, data_file in enumerate(data_files):
        if DEBUG and seq_idx <= 25:
            continue
        data_path = osp.join(data_dir, data_file)
        timestamp_debug = []

        abs_path = os.path.abspath(data_path)
        save_data_path = os.path.abspath(save_data_path)
        logger.info(f'Loading data from {abs_path}')

        # Load the data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        for step_idx, sensor_msg in enumerate(data.sensorMessages):
            if DEBUG and step_idx <= 60:
                continue
            total_count += 1
            logger.info(f'Processing {step_idx}th sensor message in sequence {seq_idx}')

            # TODO: add timestamp
            obs_dict = data_processing_manager.convert_sensor_msg_to_obs_dict(sensor_msg)
            timestamp_arrays.append(sensor_msg.timestamp)
            timestamp_debug.append(sensor_msg.timestamp)

            left_robot_tcp_pose_arrays.append(obs_dict['left_robot_tcp_pose'])
            left_robot_tcp_vel_arrays.append(obs_dict['left_robot_tcp_vel'])
            left_robot_tcp_wrench_arrays.append(obs_dict['left_robot_tcp_wrench'])
            left_robot_gripper_width_arrays.append(obs_dict['left_robot_gripper_width'])
            left_robot_gripper_force_arrays.append(obs_dict['left_robot_gripper_force'])
            right_robot_tcp_pose_arrays.append(obs_dict['right_robot_tcp_pose'])
            right_robot_tcp_vel_arrays.append(obs_dict['right_robot_tcp_vel'])
            right_robot_tcp_wrench_arrays.append(obs_dict['right_robot_tcp_wrench'])
            right_robot_gripper_width_arrays.append(obs_dict['right_robot_gripper_width'])
            right_robot_gripper_force_arrays.append(obs_dict['right_robot_gripper_force'])

            if 'left_wrist_img' in obs_dict:
                left_wrist_img_arrays.append(obs_dict['left_wrist_img'])
            if 'external_img' in obs_dict:
                external_img_arrays.append(obs_dict['external_img'])

            if DEBUG:
                # visualize external camera image
                visualize_rgb_image(sensor_msg.externalCameraRGB, 'External Camera RGB')
                # visualize left wrist camera image
                visualize_rgb_image(sensor_msg.leftWristCameraRGB, 'Left Wrist Camera RGB')
                # visualize right wrist camera image
                visualize_rgb_image(sensor_msg.rightWristCameraRGB, 'Right Wrist Camera RGB')

                logger.debug(f'left_robot_tcp_pose: {obs_dict["left_robot_tcp_pose"]}, '
                             f'right_robot_tcp_pose: {obs_dict["right_robot_tcp_pose"]}')

            del sensor_msg, obs_dict
        _ = np.mean(np.diff(np.array(timestamp_debug)))
        logger.info(f'fps of episode {seq_idx} is {1./_:.2f}')
        episode_fps.append(1./_)
        episode_ends_arrays.append(total_count)

    # Convert lists to arrays
    external_img_arrays = np.stack(external_img_arrays, axis=0)
    left_wrist_img_arrays = np.stack(left_wrist_img_arrays, axis=0)
    
    episode_ends_arrays = np.array(episode_ends_arrays)
    
    timestamp_arrays = np.array(timestamp_arrays)
    left_robot_tcp_pose_arrays = np.stack(left_robot_tcp_pose_arrays, axis=0)
    left_robot_tcp_vel_arrays = np.stack(left_robot_tcp_vel_arrays, axis=0)
    left_robot_tcp_wrench_arrays = np.stack(left_robot_tcp_wrench_arrays, axis=0)
    left_robot_gripper_width_arrays = np.stack(left_robot_gripper_width_arrays, axis=0)
    left_robot_gripper_force_arrays = np.stack(left_robot_gripper_force_arrays, axis=0)
    right_robot_tcp_pose_arrays = np.stack(right_robot_tcp_pose_arrays, axis=0)
    right_robot_tcp_vel_arrays = np.stack(right_robot_tcp_vel_arrays, axis=0)
    right_robot_tcp_wrench_arrays = np.stack(right_robot_tcp_wrench_arrays, axis=0)
    right_robot_gripper_width_arrays = np.stack(right_robot_gripper_width_arrays, axis=0)
    right_robot_gripper_force_arrays = np.stack(right_robot_gripper_force_arrays, axis=0)


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
        external_img_arrays = external_img_arrays[keep_indices]
        left_wrist_img_arrays = left_wrist_img_arrays[keep_indices]


        left_robot_tcp_pose_arrays = left_robot_tcp_pose_arrays[keep_indices]
        left_robot_tcp_vel_arrays = left_robot_tcp_vel_arrays[keep_indices]
        left_robot_tcp_wrench_arrays = left_robot_tcp_wrench_arrays[keep_indices]
        left_robot_gripper_width_arrays = left_robot_gripper_width_arrays[keep_indices]
        left_robot_gripper_force_arrays = left_robot_gripper_force_arrays[keep_indices]
        right_robot_tcp_pose_arrays = right_robot_tcp_pose_arrays[keep_indices]
        right_robot_tcp_vel_arrays = right_robot_tcp_vel_arrays[keep_indices]
        right_robot_tcp_wrench_arrays = right_robot_tcp_wrench_arrays[keep_indices]
        right_robot_gripper_width_arrays = right_robot_gripper_width_arrays[keep_indices]
        right_robot_gripper_force_arrays = right_robot_gripper_force_arrays[keep_indices]

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
    if TEMPORAL_UPSAMPLE_RATIO > 0:
        timestamp_arrays_new = []
        external_img_arrays_new = []
        left_wrist_img_arrays_new = []
        right_wrist_img_arrays_new = []
        # robot state arrays
        left_robot_tcp_pose_arrays_new = []
        left_robot_tcp_vel_arrays_new = []
        left_robot_tcp_wrench_arrays_new = []
        left_robot_gripper_width_arrays_new = []
        left_robot_gripper_force_arrays_new = []
        right_robot_tcp_pose_arrays_new = []
        right_robot_tcp_vel_arrays_new = []
        right_robot_tcp_wrench_arrays_new = []
        right_robot_gripper_width_arrays_new = []
        right_robot_gripper_force_arrays_new = []

        episode_ends_array_new = []

        current_episode_start = 0
        total_count_new = 0
        epi_num = 0

        # Process each episode separately
        for episode_end, original_fps in zip(episode_ends_arrays, episode_fps):
            # Get indices for current episode
            episode_indices = np.arange(current_episode_start, episode_end)
            # Keep first and last frame of each episode, downsample middle frames
            middle_indices = episode_indices[1:-1]
            timestamp_arrays_new.append(timestamp_arrays[episode_indices[0]])
            left_robot_tcp_pose_arrays_new.append(left_robot_tcp_pose_arrays[episode_indices[0]])
            left_robot_tcp_vel_arrays_new.append(left_robot_tcp_vel_arrays[episode_indices[0]])
            left_robot_tcp_wrench_arrays_new.append(left_robot_tcp_wrench_arrays[episode_indices[0]])
            left_robot_gripper_width_arrays_new.append(left_robot_gripper_width_arrays[episode_indices[0]])
            left_robot_gripper_force_arrays_new.append(left_robot_gripper_force_arrays[episode_indices[0]])
            right_robot_tcp_pose_arrays_new.append(right_robot_tcp_pose_arrays[episode_indices[0]])
            right_robot_tcp_vel_arrays_new.append(right_robot_tcp_vel_arrays[episode_indices[0]])
            right_robot_tcp_wrench_arrays_new.append(right_robot_tcp_wrench_arrays[episode_indices[0]])
            right_robot_gripper_width_arrays_new.append(right_robot_gripper_width_arrays[episode_indices[0]])
            right_robot_gripper_force_arrays_new.append(right_robot_gripper_force_arrays[episode_indices[0]])
            external_img_arrays_new.append(external_img_arrays[episode_indices[0]])
            left_wrist_img_arrays_new.append(left_wrist_img_arrays[episode_indices[0]])
            # Interpolate the data
            interpolated= interpolate_common(timestamp_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)
            timestamp_arrays_new = np.concatenate([timestamp_arrays_new, interpolated], axis=0).tolist()
            total_count_new += len(interpolated) + 2
            episode_ends_array_new.append(total_count_new)
            left_robot_gripper_force_arrays_new = np.concatenate([left_robot_gripper_force_arrays_new, interpolate_common(left_robot_gripper_force_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            left_robot_tcp_pose_arrays_new = np.concatenate([left_robot_tcp_pose_arrays_new, interpolate_tcp_pose(left_robot_tcp_pose_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            left_robot_tcp_vel_arrays_new = np.concatenate([left_robot_tcp_vel_arrays_new, interpolate_common(left_robot_tcp_vel_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            left_robot_tcp_wrench_arrays_new = np.concatenate([left_robot_tcp_wrench_arrays_new, interpolate_common(left_robot_tcp_wrench_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            left_robot_gripper_width_arrays_new = np.concatenate([left_robot_gripper_width_arrays_new, interpolate_common(left_robot_gripper_width_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            right_robot_gripper_force_arrays_new = np.concatenate([right_robot_gripper_force_arrays_new, interpolate_common(right_robot_gripper_force_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            right_robot_tcp_pose_arrays_new = np.concatenate([right_robot_tcp_pose_arrays_new, interpolate_tcp_pose(right_robot_tcp_pose_arrays[middle_indices],    
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            right_robot_tcp_vel_arrays_new = np.concatenate([right_robot_tcp_vel_arrays_new, interpolate_common(right_robot_tcp_vel_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            right_robot_tcp_wrench_arrays_new = np.concatenate([right_robot_tcp_wrench_arrays_new, interpolate_common(right_robot_tcp_wrench_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            right_robot_gripper_width_arrays_new = np.concatenate([right_robot_gripper_width_arrays_new, interpolate_common(right_robot_gripper_width_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps)], axis=0).tolist()
            external_img_arrays_new.extend(interpolate_images(external_img_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps))
            left_wrist_img_arrays_new.extend(interpolate_images(left_wrist_img_arrays[middle_indices],
                                                                        original_fps=original_fps,
                                                                        target_fps=TEMPORAL_UPSAMPLE_RATIO * original_fps))
            # Append the last frame of the episode
            timestamp_arrays_new.append(timestamp_arrays[episode_indices[-1]])
            left_robot_tcp_pose_arrays_new.append(left_robot_tcp_pose_arrays[episode_indices[-1]])
            left_robot_tcp_vel_arrays_new.append(left_robot_tcp_vel_arrays[episode_indices[-1]])
            left_robot_tcp_wrench_arrays_new.append(left_robot_tcp_wrench_arrays[episode_indices[-1]])
            left_robot_gripper_width_arrays_new.append(left_robot_gripper_width_arrays[episode_indices[-1]])
            left_robot_gripper_force_arrays_new.append(left_robot_gripper_force_arrays[episode_indices[-1]])
            right_robot_tcp_pose_arrays_new.append(right_robot_tcp_pose_arrays[episode_indices[-1]])
            right_robot_tcp_vel_arrays_new.append(right_robot_tcp_vel_arrays[episode_indices[-1]])
            right_robot_tcp_wrench_arrays_new.append(right_robot_tcp_wrench_arrays[episode_indices[-1]])
            right_robot_gripper_width_arrays_new.append(right_robot_gripper_width_arrays[episode_indices[-1]])
            right_robot_gripper_force_arrays_new.append(right_robot_gripper_force_arrays[episode_indices[-1]])
            external_img_arrays_new.append(external_img_arrays[episode_indices[-1]])
            left_wrist_img_arrays_new.append(left_wrist_img_arrays[episode_indices[-1]])
            logger.info(f'Interpolated {len(middle_indices)} frames in episode {epi_num} to {len(interpolated) + 2} frames, upsample ratio is {(len(interpolated) + 2) / len(middle_indices)}')
            epi_num += 1
            current_episode_start = episode_end

        # Convert lists to arrays
        external_img_arrays = np.stack(external_img_arrays_new, axis=0)
        left_wrist_img_arrays = np.stack(left_wrist_img_arrays_new, axis=0)
        
        episode_ends_arrays = np.array(episode_ends_array_new)
        
        timestamp_arrays = np.array(timestamp_arrays_new)
        left_robot_tcp_pose_arrays = np.stack(left_robot_tcp_pose_arrays_new, axis=0)
        left_robot_tcp_vel_arrays = np.stack(left_robot_tcp_vel_arrays_new, axis=0)
        left_robot_tcp_wrench_arrays = np.stack(left_robot_tcp_wrench_arrays_new, axis=0)
        left_robot_gripper_width_arrays = np.stack(left_robot_gripper_width_arrays_new, axis=0)
        left_robot_gripper_force_arrays = np.stack(left_robot_gripper_force_arrays_new, axis=0)
        right_robot_tcp_pose_arrays = np.stack(right_robot_tcp_pose_arrays_new, axis=0)
        right_robot_tcp_vel_arrays = np.stack(right_robot_tcp_vel_arrays_new, axis=0)
        right_robot_tcp_wrench_arrays = np.stack(right_robot_tcp_wrench_arrays_new, axis=0)
        right_robot_gripper_width_arrays = np.stack(right_robot_gripper_width_arrays_new, axis=0)
        right_robot_gripper_force_arrays = np.stack(right_robot_gripper_force_arrays_new, axis=0)

    if ACTION_DIM == 4: # (left_tcp_x, left_tcp_y, left_tcp_z, left_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays[:, :3], left_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 8: # (left_tcp_x, left_tcp_y, left_tcp_z, right_tcp_x, right_tcp_y, right_tcp_z, left_gripper_width, right_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays[:, :3], right_robot_tcp_pose_arrays[:, :3],
                                       left_robot_gripper_width_arrays, right_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 10: # (left_tcp_x, left_tcp_y, left_tcp_z, left_6d_rotation, left_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays, left_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 20: # (left_tcp_x, left_tcp_y, left_tcp_z, left_6d_rotation, right_tcp_x, right_tcp_y, right_tcp_z, right_6d_rotation, left_gripper_width, right_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays, right_robot_tcp_pose_arrays,
                                       left_robot_gripper_width_arrays, right_robot_gripper_width_arrays], axis=-1)
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

    if ACTION_DIM == 4: # (left_tcp_x, left_tcp_y, left_tcp_z, left_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays[:, :3], left_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 8: # (left_tcp_x, left_tcp_y, left_tcp_z, right_tcp_x, right_tcp_y, right_tcp_z, left_gripper_width, right_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays[:, :3], right_robot_tcp_pose_arrays[:, :3],
                                       left_robot_gripper_width_arrays, right_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 10: # (left_tcp_x, left_tcp_y, left_tcp_z, left_6d_rotation, left_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays, left_robot_gripper_width_arrays], axis=-1)
    elif ACTION_DIM == 20: # (left_tcp_x, left_tcp_y, left_tcp_z, left_6d_rotation, right_tcp_x, right_tcp_y, right_tcp_z, right_6d_rotation, left_gripper_width, right_gripper_width)
        state_arrays = np.concatenate([left_robot_tcp_pose_arrays, right_robot_tcp_pose_arrays,
                                       left_robot_gripper_width_arrays, right_robot_gripper_width_arrays], axis=-1)
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
    external_img_chunk_size = (100, external_img_arrays.shape[1], external_img_arrays.shape[2], external_img_arrays.shape[3])
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
    zarr_data.create_dataset('left_robot_tcp_vel', data=left_robot_tcp_vel_arrays, chunks=(10000, 6), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('left_robot_tcp_wrench', data=left_robot_tcp_wrench_arrays, chunks=(10000, 6), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('left_robot_gripper_width', data=left_robot_gripper_width_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('left_robot_gripper_force', data=left_robot_gripper_force_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_tcp_pose', data=right_robot_tcp_pose_arrays, chunks=(10000, 9), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_tcp_vel', data=right_robot_tcp_vel_arrays, chunks=(10000, 6), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_tcp_wrench', data=right_robot_tcp_wrench_arrays, chunks=(10000, 6), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_gripper_width', data=right_robot_gripper_width_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)
    zarr_data.create_dataset('right_robot_gripper_force', data=right_robot_gripper_force_arrays, chunks=(10000, 1), dtype='float32', overwrite=True,
                                compressor=compressor)

    zarr_data.create_dataset('target', data=state_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True,
                             compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(10000,), dtype='int64', overwrite=True,
                             compressor=compressor)

    zarr_data.create_dataset('external_img', data=external_img_arrays, chunks=external_img_chunk_size, dtype='uint8', overwrite=True,
                             compressor=compressor)
    zarr_data.create_dataset('left_wrist_img', data=left_wrist_img_arrays, chunks=wrist_img_chunk_size, dtype='uint8')
    

    # print zarr data structure
    logger.info('Zarr data structure')
    logger.info(zarr_data.tree())
    logger.info(f'Total count after filtering: {action_arrays.shape[0]}')
    logger.info(f'Save data at {save_data_path}')
