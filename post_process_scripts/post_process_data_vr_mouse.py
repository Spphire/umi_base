import os
from loguru import logger
import zarr
import cv2
import numpy as np
import os.path as osp
import tarfile
from tqdm import tqdm
from typing import Optional
import json
import lz4.frame
import torch
from PIL import Image

from diffusion_policy.real_world.post_process_utils import DataPostProcessingManagerVR
from diffusion_policy.common.image_utils import center_crop_and_resize_image
from diffusion_policy.common.data_models import ActionType

# GroundedSAM imports
try:
    from groundingdino.util.inference import load_model, load_image, predict
    from segment_anything import sam_model_registry, SamPredictor
    GROUNDEDSAM_AVAILABLE = True
except ImportError:
    logger.warning("GroundedSAM not available. Install it for arm masking functionality.")
    GROUNDEDSAM_AVAILABLE = False


class HandArmSegmentor:
    """使用GroundedSAM检测和分割人手和手臂的工具类"""
    
    def __init__(
        self, 
        grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint: str = "weights/groundingdino_swint_ogc.pth",
        sam_checkpoint: str = "weights/sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化手臂分割器
        
        参数:
            grounding_dino_config: GroundingDINO配置文件路径
            grounding_dino_checkpoint: GroundingDINO权重文件路径
            sam_checkpoint: SAM权重文件路径
            sam_model_type: SAM模型类型
            device: 运行设备
        """
        if not GROUNDEDSAM_AVAILABLE:
            raise ImportError("GroundedSAM is not installed. Please install it first.")
        
        self.device = device
        
        # 加载GroundingDINO模型
        logger.info(f"Loading GroundingDINO model from {grounding_dino_checkpoint}")
        self.grounding_model = load_model(grounding_dino_config, grounding_dino_checkpoint, device=device)
        
        # 加载SAM模型
        logger.info(f"Loading SAM model from {sam_checkpoint}")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        # 检测文本提示（英文，用于检测人手和手臂）
        self.text_prompt = "hand . arm . human hand . human arm"
        self.box_threshold = 0.25
        self.text_threshold = 0.25
        
        logger.info("HandArmSegmentor initialized successfully")
    
    def segment_hand_arm(self, image: np.ndarray) -> np.ndarray:
        """
        对单张图像进行手和手臂的分割
        
        参数:
            image: 输入图像，numpy数组，shape为(H, W, 3)，BGR格式
            
        返回:
            mask: 二值mask，shape为(H, W)，手和手臂区域为True，其他为False
        """
        # 转换为PIL Image以供GroundingDINO使用
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # 使用GroundingDINO检测手和手臂
        image_source, image_tensor = load_image(image_pil)
        
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )
        
        # 如果没有检测到手或手臂，返回空mask
        if len(boxes) == 0:
            return np.zeros(image.shape[:2], dtype=bool)
        
        # 使用SAM进行精确分割
        self.sam_predictor.set_image(image_rgb)
        
        # 将boxes转换为SAM格式
        h, w = image.shape[:2]
        boxes_xyxy = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = boxes_xyxy.cpu().numpy()
        
        # 使用SAM预测masks
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            torch.from_numpy(boxes_xyxy).to(self.device), 
            image_rgb.shape[:2]
        )
        
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # 合并所有检测到的masks
        combined_mask = masks.cpu().numpy().any(axis=0).squeeze()
        
        return combined_mask
    
    def mask_hand_arm_in_image(self, image: np.ndarray, mask_value: int = 0) -> np.ndarray:
        """
        在图像中mask掉手和手臂区域
        
        参数:
            image: 输入图像，numpy数组，shape为(H, W, 3)
            mask_value: 用于填充mask区域的值，0为黑色，255为白色
            
        返回:
            masked_image: 处理后的图像
        """
        mask = self.segment_hand_arm(image)
        masked_image = image.copy()
        masked_image[mask] = mask_value
        return masked_image


def convert_data_to_zarr(
    input_dir: str,
    output_dir: str,
    temporal_downsample_ratio: int = 3,
    use_absolute_action: bool = True,
    action_type: ActionType = ActionType.head_6DOF_left_arm_6DOF_gripper_width,
    debug: bool = False,
    overwrite: bool = True,
    use_dino: bool = False,
    gripper_width_bias: float = 0.0,
    gripper_width_scale: float = 1.0,
    episode_clip_head_seconds: float = 0.0,
    episode_clip_tail_seconds: float = 0.0,
    use_hand_masking: bool = True,
    hand_mask_value: int = 0,
    grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint: str = "GroundingDINO/weights/groundingdino_swint_ogc.pth",
    sam_checkpoint: str = "GroundingDINO/weights/sam_vit_h_4b8939.pth",
    sam_model_type: str = "vit_h"
) -> str:
    """
    将VR原始数据转换为zarr格式存储（仅左臂+头部，头部图像带手臂遮罩）
    
    参数:
        input_dir (str): 输入数据目录，包含.tar.gz文件
        output_dir (str): 输出目录，用于保存zarr文件
        temporal_downsample_ratio (int): 时序降采样比例
        use_absolute_action (bool): 是否使用绝对动作值
        action_type (ActionType): 动作类型（仅支持head_6DOF_left_arm_6DOF_gripper_width）
        debug (bool): 是否开启调试模式
        overwrite (bool): 是否覆盖已存在的数据
        use_dino (bool): 是否使用DINO
        gripper_width_bias (float): 夹爪宽度偏差
        gripper_width_scale (float): 夹爪宽度缩放比例
        episode_clip_head_seconds (float): 剪掉每个episode开头的秒数
        episode_clip_tail_seconds (float): 剪掉每个episode结尾的秒数
        use_hand_masking (bool): 是否对头部图像进行手臂遮罩
        hand_mask_value (int): 遮罩值，0为黑色，255为白色
        grounding_dino_config (str): GroundingDINO配置文件路径
        grounding_dino_checkpoint (str): GroundingDINO权重文件路径
        sam_checkpoint (str): SAM权重文件路径
        sam_model_type (str): SAM模型类型
        
    返回:
        str: 保存的zarr文件路径
    """
    # 验证action_type
    if action_type != ActionType.head_6DOF_left_arm_6DOF_gripper_width:
        raise ValueError(
            f"This script only supports ActionType.head_6DOF_left_arm_6DOF_gripper_width, "
            f"but got {action_type.name}"
        )
    
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
            import shutil
            shutil.rmtree(save_data_path)
    
    # 初始化手臂分割器（如果需要）
    hand_segmentor = None
    if use_hand_masking:
        if not GROUNDEDSAM_AVAILABLE:
            logger.warning(
                "GroundedSAM not available, skipping hand masking. "
                "Install it with: pip install groundingdino-py segment-anything"
            )
            use_hand_masking = False
        else:
            try:
                hand_segmentor = HandArmSegmentor(
                    grounding_dino_config=grounding_dino_config,
                    grounding_dino_checkpoint=grounding_dino_checkpoint,
                    sam_checkpoint=sam_checkpoint,
                    sam_model_type=sam_model_type
                )
            except Exception as e:
                logger.error(f"Failed to initialize HandArmSegmentor: {e}")
                logger.warning("Proceeding without hand masking")
                use_hand_masking = False
    
    # 创建数据处理管理器
    data_processing_manager = DataPostProcessingManagerVR(use_6d_rotation=True)
    
    # 初始化数据数组
    timestamp_arrays = []
    left_wrist_img_arrays = []
    left_robot_tcp_pose_arrays = []
    left_robot_gripper_width_arrays = []
    left_eye_tcp_pose_arrays = []
    left_eye_img_arrays = []
    episode_ends_arrays = []
    total_count = 0
    
    # 处理所有未解压数据文件
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tar.gz')])
    logger.info(f"Found {len(data_files)} data files in {data_dir}")
    
    for seq_idx, data_file in enumerate(data_files):
        if debug and seq_idx >= 5:
            logger.info(f"[Debug] Stopping after 5 files due to debug mode")
            break
        
        data_path = osp.join(data_dir, data_file)
        abs_path = os.path.abspath(data_path)
        
        # 解压数据文件
        logger.info(f"Extracting {abs_path}...")
        try:
            with tarfile.open(abs_path, 'r:gz') as tar:
                tar.extractall(path=input_dir)
        except tarfile.ReadError:
            # Try lz4 compressed tar
            logger.info(f"Trying lz4 decompression for {abs_path}...")
            try:
                with lz4.frame.open(abs_path, 'rb') as lz4_file:
                    with tarfile.open(fileobj=lz4_file, mode='r|') as tar:
                        tar.extractall(path=input_dir)
            except Exception as e:
                logger.error(f"Failed to extract {abs_path}: {e}")
                continue
    
    # Get directories containing .bson files
    dst_paths = []
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for subfolder in subfolders:
        try:
            if any(f.endswith('.bson') for f in os.listdir(subfolder)):
                dst_paths.append(subfolder)
        except PermissionError:
            continue
    
    if not dst_paths:
        logger.warning(f"No .bson files found in subdirectories of {data_dir}")
        return save_data_path
    
    # 组织sessions
    record_sessions = {}
    for dst_path in dst_paths:
        meta_path = osp.join(dst_path, 'metadata.json')
        if not os.path.exists(meta_path):
            logger.warning(f"metadata.json not found in {dst_path}, skipping")
            continue
            
        try:
            metadata = json.load(open(meta_path, 'r'))
            uuid = metadata['uuid']
            session_uuid = metadata['parent_uuid'] if 'parent_uuid' in metadata else uuid
            if session_uuid not in record_sessions:
                record_sessions[session_uuid] = {}
            
            camera_position = metadata.get('camera_position', '')
            if camera_position == '':
                camera_position = 'head'
            record_sessions[session_uuid][camera_position] = dst_path
        except Exception as e:
            logger.error(f"Failed to read metadata from {meta_path}: {e}")
            continue
    
    # 处理每个session
    skipped_sessions = 0
    
    for session_idx, session in tqdm(enumerate(record_sessions.items()), 
                                      total=len(record_sessions),
                                      desc="Processing sessions",
                                      dynamic_ncols=True):
        try:
            obs_dict = data_processing_manager.extract_msg_to_obs_dict(
                session[1],
                clip_head_seconds=episode_clip_head_seconds,
                clip_tail_seconds=episode_clip_tail_seconds,
                use_aruco_calibration=False
            )
        except Exception as e:
            logger.error(f"Failed to extract obs_dict for session {session[0]}: {e}")
            skipped_sessions += 1
            continue
            
        if obs_dict is None:
            logger.warning(f"obs_dict is None for {session[0]}")
            skipped_sessions += 1
            continue
        
        # 检查数据完整性
        has_left = 'left_robot_tcp_pose' in obs_dict and 'left_robot_gripper_width' in obs_dict
        has_head = 'left_eye_tcp_pose' in obs_dict and 'left_eye_img' in obs_dict
        
        if not (has_left and has_head):
            skipped_sessions += 1
            missing_parts = []
            if not has_left:
                missing_parts.append('left arm')
            if not has_head:
                missing_parts.append('head')
            logger.warning(
                f"Session {session[0]} skipped: requires head and left arm data, "
                f"but missing {', '.join(missing_parts)} data"
            )
            continue
        
        # 收集timestamp
        timestamp_arrays.append(obs_dict['timestamp'])
        
        # 处理左臂TCP pose（无坐标转换）
        left_robot_tcp_pose_arrays.append(obs_dict['left_robot_tcp_pose'])
        
        total_count += len(obs_dict['timestamp'])
        episode_ends_arrays.append(total_count)
        
        # 处理左臂夹爪宽度
        gripper_width = obs_dict['left_robot_gripper_width'].copy()
        # 平滑处理异常值
        for i in range(1, len(gripper_width) - 2):
            if abs(gripper_width[i] - gripper_width[i-1]) > 0.15:
                gripper_width[i] = (gripper_width[i-1] + gripper_width[i+2]) / 2
        left_robot_gripper_width_arrays.append(gripper_width)
        
        # 补齐gripper_width数据
        gripper_width_pad_cnt = 0
        while len(left_robot_gripper_width_arrays[-1]) < len(left_robot_tcp_pose_arrays[-1]):
            left_robot_gripper_width_arrays[-1] = np.concatenate([
                left_robot_gripper_width_arrays[-1],
                left_robot_gripper_width_arrays[-1][-1][np.newaxis, :]
            ])
            gripper_width_pad_cnt += 1
        if gripper_width_pad_cnt > 0:
            logger.warning(f"Gripper width data padded {gripper_width_pad_cnt} times for session {session[0]}")
        
        # 处理左臂腕部图像
        if 'left_wrist_img' in obs_dict:
            # 保存第一张图片到.cache
            if not os.path.exists('.cache'):
                os.makedirs('.cache')
            first_image_path = '.cache/left_wrist_img_0.png'
            cv2.imwrite(first_image_path, obs_dict['left_wrist_img'][0])
            logger.info(f"First left wrist image saved to {first_image_path}")
            
            if use_dino:
                processed_images = []
                for img in obs_dict['left_wrist_img']:
                    processed_img = center_crop_and_resize_image(img)
                    processed_images.append(processed_img)
                left_wrist_img_arrays.append(np.array(processed_images))
            else:
                left_wrist_img_arrays.append(np.array(obs_dict['left_wrist_img']))
        
        # 处理头部TCP pose（无坐标转换）
        left_eye_tcp_pose_arrays.append(obs_dict['left_eye_tcp_pose'])
        
        # 处理头部图像（应用手臂遮罩）
        if 'left_eye_img' in obs_dict:
            # 保存第一张原始图片
            if not os.path.exists('.cache'):
                os.makedirs('.cache')
            first_image_path = '.cache/left_eye_img_0_raw.png'
            cv2.imwrite(first_image_path, obs_dict['left_eye_img'][0])
            logger.info(f"First left eye image (raw) saved to {first_image_path}")
            
            processed_images = []
            
            # 应用手臂遮罩
            if use_hand_masking and hand_segmentor is not None:
                logger.info(f"Applying hand/arm masking to {len(obs_dict['left_eye_img'])} frames...")
                for idx, img in enumerate(tqdm(obs_dict['left_eye_img'], 
                                                desc=f"Masking session {session_idx+1}",
                                                leave=False)):
                    try:
                        masked_img = hand_segmentor.mask_hand_arm_in_image(img, mask_value=hand_mask_value)
                        
                        # 保存第一张masked图片以便检查
                        if idx == 0:
                            first_masked_path = '.cache/left_eye_img_0_masked.png'
                            cv2.imwrite(first_masked_path, masked_img)
                            logger.info(f"First left eye image (masked) saved to {first_masked_path}")
                        
                        if use_dino:
                            masked_img = center_crop_and_resize_image(masked_img, crop=False)
                        processed_images.append(masked_img)
                    except Exception as e:
                        logger.error(f"Failed to mask frame {idx} in session {session[0]}: {e}")
                        # 如果遮罩失败，使用原图
                        img_to_use = img
                        if use_dino:
                            img_to_use = center_crop_and_resize_image(img, crop=False)
                        processed_images.append(img_to_use)
            else:
                # 不使用遮罩
                for img in obs_dict['left_eye_img']:
                    if use_dino:
                        img = center_crop_and_resize_image(img, crop=False)
                    processed_images.append(img)
            
            left_eye_img_arrays.append(np.array(processed_images))
    
    if skipped_sessions > 0:
        logger.warning(
            f"{skipped_sessions} session(s) skipped due to incomplete data "
            f"(missing head or left arm data)."
        )
    
    # 转换列表为数组
    episode_ends_arrays = np.array(episode_ends_arrays)
    timestamp_arrays = np.vstack(timestamp_arrays)
    left_robot_tcp_pose_arrays = np.vstack(left_robot_tcp_pose_arrays)
    left_robot_gripper_width_arrays = np.vstack(left_robot_gripper_width_arrays)
    left_robot_gripper_width_arrays = (left_robot_gripper_width_arrays + gripper_width_bias) * gripper_width_scale
    
    if left_wrist_img_arrays:
        left_wrist_img_arrays = np.vstack(left_wrist_img_arrays)
    
    if left_eye_tcp_pose_arrays:
        left_eye_tcp_pose_arrays = np.vstack(left_eye_tcp_pose_arrays)
    if left_eye_img_arrays:
        left_eye_img_arrays = np.vstack(left_eye_img_arrays)
    
    logger.info(f"Total episodes: {len(episode_ends_arrays)}")
    logger.info(f"Total frames: {len(timestamp_arrays)}")
    
    # 构建状态数组
    state_arrays = np.concatenate([
        left_robot_tcp_pose_arrays,
        left_robot_gripper_width_arrays,
        left_eye_tcp_pose_arrays,
    ], axis=-1)
    
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
        left_wrist_img_arrays,
        left_eye_tcp_pose_arrays,
        left_eye_img_arrays
    )
    
    # 打印数据结构信息
    logger.info('Zarr data structure:')
    logger.info(zarr_data.tree())
    logger.info(f'Total count after filtering: {action_arrays.shape[0]}')
    logger.info(f'Save data at {save_data_path}')
    
    return save_data_path


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
    left_wrist_img_arrays: Optional[np.ndarray] = None,
    left_eye_tcp_pose_arrays: Optional[np.ndarray] = None,
    left_eye_img_arrays: Optional[np.ndarray] = None
) -> tuple:
    """创建zarr存储"""
    zarr_root = zarr.group(save_data_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    
    # 计算chunk大小
    action_chunk_size = (10000, action_arrays.shape[1])
    if left_wrist_img_arrays is not None and len(left_wrist_img_arrays) > 0:
        wrist_img_chunk_size = (100, *left_wrist_img_arrays.shape[1:])
    elif left_eye_img_arrays is not None and len(left_eye_img_arrays) > 0:
        wrist_img_chunk_size = (100, *left_eye_img_arrays.shape[1:])
    else:
        wrist_img_chunk_size = (100, 480, 640, 3)  # 默认大小
    
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
    
    if left_wrist_img_arrays is not None and len(left_wrist_img_arrays) > 0:
        zarr_data.create_dataset('left_wrist_img', data=left_wrist_img_arrays,
                               chunks=wrist_img_chunk_size, dtype='uint8')
    
    if left_eye_tcp_pose_arrays is not None and len(left_eye_tcp_pose_arrays) > 0:
        zarr_data.create_dataset('left_eye_tcp_pose', data=left_eye_tcp_pose_arrays,
                               chunks=(10000, 9), dtype='float32',
                               overwrite=True, compressor=compressor)
    
    if left_eye_img_arrays is not None and len(left_eye_img_arrays) > 0:
        zarr_data.create_dataset('left_eye_img', data=left_eye_img_arrays,
                               chunks=wrist_img_chunk_size, dtype='uint8')
    
    return zarr_data, zarr_meta


if __name__ == '__main__':
    # 示例使用
    input_dir = '/mnt/data/shenyibo/workspace/umi_base/.cache/targz_blockq3_1-28'
    output_dir = '/mnt/data/shenyibo/workspace/umi_base/.cache/blockq3_1-28_masked'
    debug = False  # 设置为True以进行调试（只处理前5个文件）
    temporal_downsample_ratio = 1  # 设置时序降采样比例
    use_absolute_action = True  # 使用绝对动作
    action_type = ActionType.head_6DOF_left_arm_6DOF_gripper_width  # 设置动作类型
    overwrite = True  # 是否覆盖已有数据
    use_dino = False  # 是否使用DINO
    gripper_width_bias = 0.0  # 设置夹爪宽度偏差
    gripper_width_scale = 1.0  # 设置夹爪宽度缩放比例
    
    # 手臂遮罩相关参数
    use_hand_masking = True  # 是否使用手臂遮罩
    hand_mask_value = 0  # 遮罩值，0为黑色，255为白色
    
    # GroundedSAM模型路径（需要根据实际情况修改）
    grounding_dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint = "weights/groundingdino_swint_ogc.pth"
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    
    zarr_path = convert_data_to_zarr(
        input_dir=input_dir,
        output_dir=output_dir,
        temporal_downsample_ratio=temporal_downsample_ratio,
        use_absolute_action=use_absolute_action,
        action_type=action_type,
        debug=debug,
        overwrite=overwrite,
        use_dino=use_dino,
        gripper_width_bias=gripper_width_bias,
        gripper_width_scale=gripper_width_scale,
        episode_clip_head_seconds=0.3,
        episode_clip_tail_seconds=0.3,
        use_hand_masking=use_hand_masking,
        hand_mask_value=hand_mask_value,
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type
    )
    
    print(f"Data saved to {zarr_path}")
