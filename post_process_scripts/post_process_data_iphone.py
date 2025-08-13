import os
from loguru import logger
import zarr
import cv2
import numpy as np
import os.path as osp
import tarfile
from tqdm import tqdm

from diffusion_policy.real_world.post_process_utils import DataPostProcessingManageriPhone
from diffusion_policy.common.image_utils import center_pad_and_resize_image, center_crop_and_resize_image
from diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
from diffusion_policy.common.space_utils import pose_3d_9d_to_homo_matrix_batch, homo_matrix_to_pose_9d_batch

def convert_data_to_zarr(
    input_dir: str,
    output_dir: str,
    temporal_downsample_ratio: int = 3,
    use_absolute_action: bool = True,
    action_dim: int = 10,
    debug: bool = False,
    overwrite: bool = True,
    use_dino: bool = False,
    gripper_width_bias: float = 0.0,
    gripper_width_scale: float = 0.1,
    tcp_transform: np.ndarray = np.eye(4, dtype=np.float32),
    image_mask_path: str = ''
) -> str:
    """
    将原始数据转换为zarr格式存储。

    参数:
        input_dir (str): 输入数据目录，包含.tar.gz文件
        output_dir (str): 输出目录，用于保存zarr文件
        temporal_downsample_ratio (int): 时序降采样比例
        use_absolute_action (bool): 是否使用绝对动作值
        action_dim (int): 动作维度 (4或10)
        debug (bool): 是否开启调试模式
        overwrite (bool): 是否覆盖已存在的数据
        use_dino (bool): 是否使用DINO
        gripper_width_bias (float): 夹爪宽度偏差
        gripper_width_scale (float): 夹爪宽度缩放比例

    返回:
        str: 保存的zarr文件路径
    """
    data_dir = input_dir
    save_data_dir = output_dir
    save_data_path = osp.join(save_data_dir, f'replay_buffer.zarr')
    
    # 创建保存目录
    os.makedirs(save_data_dir, exist_ok=True)
    
    # 检查是否存在已有数据
    if os.path.exists(save_data_path):
        if not overwrite:
            logger.info(f'Data already exists at {save_data_path}')
            return save_data_path
        else:
            logger.warning(f'Overwriting {save_data_path}')
            os.system(f'rm -rf {save_data_path}')

    # 创建数据处理管理器
    data_processing_manager = DataPostProcessingManageriPhone(use_6d_rotation=True)

    # 初始化数据数组
    timestamp_arrays = []
    left_wrist_img_arrays = []
    left_robot_tcp_pose_arrays = []
    left_robot_gripper_width_arrays = []
    episode_ends_arrays = []
    total_count = 0

    # 处理所有未解压数据文件
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tar.gz')])
    for seq_idx, data_file in enumerate(data_files):
        if debug and seq_idx <= 5:
            continue
            
        data_path = osp.join(data_dir, data_file)
        abs_path = os.path.abspath(data_path)
        dst_path = abs_path.split('.tar.gz')[0]
        
        # 解压数据文件
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            logger.info(f"Extracting {abs_path}...")
            with tarfile.open(abs_path, 'r:gz') as tar:
                tar.extractall(path=dst_path)
    
    # Get directories containing .bson files
    dst_paths = []
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for subfolder in subfolders:
        if any(f.endswith('.bson') for f in os.listdir(subfolder)):
            dst_paths.append(subfolder)

    if not dst_paths:
        logger.warning(f"No .bson files found in subdirectories of {data_dir}")
        return save_data_path

    # Process each path containing .bson files
    for dst_path in tqdm(dst_paths, dynamic_ncols=True):
        # 提取观测数据
        obs_dict = data_processing_manager.extract_msg_to_obs_dict(dst_path)
        if obs_dict is None:
            logger.warning(f"obs_dict is None for {dst_path}")
            continue
            
        # 收集数据
        timestamp_arrays.append(obs_dict['timestamp'])

        # Transform robot TCP pose
        # if np.eye == tcp_transform:
        #     left_robot_tcp_pose_arrays.append(obs_dict['left_robot_tcp_pose'])
        # else:
        from tests.test_tcp_translation import get_tcp_transforms
        # tcp_transform = get_tcp_transforms()
        for i in range(len(obs_dict['left_robot_tcp_pose'])):
            pose_array = obs_dict['left_robot_tcp_pose'][i][np.newaxis, :]
            pose_homo_matrix = pose_3d_9d_to_homo_matrix_batch(pose_array)
            transformed_tcp_matrix = tcp_transform @ pose_homo_matrix
            transformed_9d_pose = homo_matrix_to_pose_9d_batch(transformed_tcp_matrix).squeeze()
            left_robot_tcp_pose_arrays.append(transformed_9d_pose)

        total_count += len(obs_dict['timestamp'])
        episode_ends_arrays.append(total_count)

        gripper_width = obs_dict['left_robot_gripper_width']
        for i in range(1, len(gripper_width) - 2):
            if abs(gripper_width[i] - gripper_width[i-1]) > 0.15:
                gripper_width[i] = (gripper_width[i-1] + gripper_width[i+2]) / 2
        left_robot_gripper_width_arrays.append(gripper_width)
        
        gripper_width_abs_cnt = 0
        while len(left_robot_gripper_width_arrays[-1]) < len(left_robot_tcp_pose_arrays[-1]):
            left_robot_gripper_width_arrays[-1] = np.concatenate([
                left_robot_gripper_width_arrays[-1], 
                left_robot_gripper_width_arrays[-1][-1][np.newaxis, :]
            ])
            gripper_width_abs_cnt += 1
        if gripper_width_abs_cnt > 0:
            logger.warning(f"Gripper width data padded {gripper_width_abs_cnt} times for {dst_path}")

        # 处理图像数据
        # 保存第一张图片到.cache/left_wrist_img_0.png
        if not os.path.exists('.cache'):
            os.makedirs('.cache')
        first_image_path = '.cache/left_wrist_img_0.png'
        cv2.imwrite(first_image_path, obs_dict['left_wrist_img'][0])
        logger.info(f"First image saved to {first_image_path}")
        if use_dino:
            processed_images = []
            for img in obs_dict['left_wrist_img']:
                processed_img = center_crop_and_resize_image(img)
                processed_images.append(processed_img)
            left_wrist_img_arrays.append(np.array(processed_images))
        else:
            left_wrist_img_arrays.append(np.array(obs_dict['left_wrist_img']))

    # 转换列表为数组
    left_wrist_img_arrays = np.vstack(left_wrist_img_arrays)
    episode_ends_arrays = np.array(episode_ends_arrays)
    timestamp_arrays = np.vstack(timestamp_arrays)
    left_robot_tcp_pose_arrays = np.vstack(left_robot_tcp_pose_arrays)
    left_robot_gripper_width_arrays = np.vstack(left_robot_gripper_width_arrays)
    left_robot_gripper_width_arrays = (left_robot_gripper_width_arrays + gripper_width_bias) * gripper_width_scale

    # 时序降采样处理
    if temporal_downsample_ratio > 1:
        (
            timestamp_arrays,
            left_wrist_img_arrays,
            left_robot_tcp_pose_arrays,
            left_robot_gripper_width_arrays,
            episode_ends_arrays
        ) = downsample_temporal_data(
            temporal_downsample_ratio,
            timestamp_arrays,
            left_wrist_img_arrays,
            left_robot_tcp_pose_arrays,
            left_robot_gripper_width_arrays,
            episode_ends_arrays
        )

    # 构建状态数组
    if action_dim == 4:
        state_arrays = np.concatenate([
            left_robot_tcp_pose_arrays[:, :3], 
            left_robot_gripper_width_arrays
        ], axis=-1)
    elif action_dim == 10:
        state_arrays = np.concatenate([
            left_robot_tcp_pose_arrays,
            left_robot_gripper_width_arrays
        ], axis=-1)
    else:
        raise NotImplementedError(f"Unsupported action_dim: {action_dim}")

    # 构建动作数组
    if use_absolute_action:
        action_arrays = create_absolute_actions(state_arrays, episode_ends_arrays)
    else:
        raise NotImplementedError("Only absolute actions are supported")

    # 创建zarr存储
    zarr_data, zarr_meta = create_zarr_storage(
        save_data_path,
        timestamp_arrays,
        left_robot_tcp_pose_arrays,
        left_robot_gripper_width_arrays,
        state_arrays,
        action_arrays,
        episode_ends_arrays,
        left_wrist_img_arrays
    )

    # 打印数据结构信息
    logger.info('Zarr data structure')
    logger.info(zarr_data.tree())
    logger.info(f'Total count after filtering: {action_arrays.shape[0]}')
    logger.info(f'Save data at {save_data_path}')

    return save_data_path

def downsample_temporal_data(
    downsample_ratio: int,
    timestamp_arrays: np.ndarray,
    left_wrist_img_arrays: np.ndarray,
    left_robot_tcp_pose_arrays: np.ndarray,
    left_robot_gripper_width_arrays: np.ndarray,
    episode_ends_arrays: np.ndarray
) -> tuple:
    """时序降采样处理函数"""
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
    
    # 降采样所有数组
    timestamp_arrays = timestamp_arrays[keep_indices]
    left_wrist_img_arrays = left_wrist_img_arrays[keep_indices]
    left_robot_tcp_pose_arrays = left_robot_tcp_pose_arrays[keep_indices]
    left_robot_gripper_width_arrays = left_robot_gripper_width_arrays[keep_indices]
    
    # 重新计算episode_ends
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
    """创建绝对动作数组"""
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
) -> tuple:
    """创建zarr存储"""
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    
    # 计算chunk大小
    wrist_img_chunk_size = (100, *left_wrist_img_arrays.shape[1:])
    action_chunk_size = (10000, action_arrays.shape[1])
    
    # 创建压缩器
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # 创建数据集
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
    
    return zarr_data, zarr_meta

if __name__ == '__main__':
    # 示例使用
    # tag = 'real_pick_and_place_coffee_iphone'
    input_dir = ''
    output_dir = ''
    debug = True  # 设置为True以进行调试
    temporal_downsample_ratio = 3  # 设置时序降采样比例
    use_absolute_action = True  # 使用绝对动作
    action_dim = 10  # 设置动作维度
    overwrite = True  # 是否覆盖已有数据
    use_dino = False  # 是否使用DINO
    gripper_width_bias = 0.0  # 设置夹爪宽度偏差
    gripper_width_scale = 0.1  # 设置夹爪宽度缩放比例
    
    zarr_path = convert_data_to_zarr(
        input_dir=input_dir,
        output_dir=output_dir,
        temporal_downsample_ratio=temporal_downsample_ratio,
        use_absolute_action=use_absolute_action,
        action_dim=action_dim,
        debug=debug,
        overwrite=overwrite,
        use_dino=use_dino,
        gripper_width_bias=gripper_width_bias,
        gripper_width_scale=gripper_width_scale
    )