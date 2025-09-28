import zarr
import numpy as np
import rerun as rr
import rclpy
import threading
import time
from rclpy.executors import MultiThreadedExecutor
from loguru import logger
from diffusion_policy.common.space_utils import pose_3d_9d_to_homo_matrix_batch, matrix4x4_to_pose_6d, homo_matrix_to_pose_9d_batch
from diffusion_policy.common.action_utils import absolute_actions_to_relative_actions, relative_actions_to_absolute_actions

from omegaconf import DictConfig, ListConfig
from typing import Dict, Tuple, Union, Optional, List
from diffusion_policy.env.real_bimanual.real_env import RealRobotEnvironment
from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms


def visualize_pose_matrices(pose_matrices: np.ndarray, 
                          entity_path: str,
                          arrow_length: float = 0.02,
                          colors: Optional[np.ndarray] = None,
                          show_coordinate_frame: bool = True) -> None:
    """
    可视化一系列用4x4矩阵表示的姿态，显示位置和朝向
    
    Args:
        pose_matrices: shape (N, 4, 4) 的4x4变换矩阵数组
        entity_path: rerun中的实体路径
        arrow_length: 箭头长度，用于表示朝向
        colors: 可选的颜色数组，shape (N, 3)，如果为None则使用红到蓝的渐变色
        show_coordinate_frame: 是否显示坐标系（xyz轴）
    """
    if pose_matrices.ndim == 2:
        pose_matrices = pose_matrices[np.newaxis, ...]
    
    num_poses = pose_matrices.shape[0]
    
    # 提取位置
    positions = pose_matrices[:, :3, 3]
    
    # 生成颜色
    if colors is None:
        colors = np.zeros((num_poses, 3), dtype=np.float32)
        t = np.linspace(0, 1, num_poses)
        colors[:, 0] = 1 - t  # 红色分量从1递减到0
        colors[:, 1] = 0      # 绿色分量保持0
        colors[:, 2] = t      # 蓝色分量从0递增到1
    
    # 记录位置点
    rr.log(f"{entity_path}/positions", rr.Points3D(positions, colors=colors))
    
    if show_coordinate_frame:
        # 为每个姿态创建坐标系箭头
        for i, pose in enumerate(pose_matrices):
            position = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            
            # X轴 - 红色
            x_axis_end = position + rotation_matrix[:, 0] * arrow_length
            rr.log(f"{entity_path}/x_axis/{i}", 
                   rr.Arrows3D(origins=[position], vectors=[rotation_matrix[:, 0] * arrow_length], 
                              colors=[[1.0, 0.0, 0.0]]))
            
            # Y轴 - 绿色
            y_axis_end = position + rotation_matrix[:, 1] * arrow_length
            rr.log(f"{entity_path}/y_axis/{i}", 
                   rr.Arrows3D(origins=[position], vectors=[rotation_matrix[:, 1] * arrow_length], 
                              colors=[[0.0, 1.0, 0.0]]))
            
            # Z轴 - 蓝色
            z_axis_end = position + rotation_matrix[:, 2] * arrow_length
            rr.log(f"{entity_path}/z_axis/{i}", 
                   rr.Arrows3D(origins=[position], vectors=[rotation_matrix[:, 2] * arrow_length], 
                              colors=[[0.0, 0.0, 1.0]]))
    else:
        # 只显示主方向（通常是Z轴方向）
        origins = positions
        vectors = pose_matrices[:, :3, 2] * arrow_length  # Z轴方向
        rr.log(f"{entity_path}/orientations", 
               rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))


def generate_gradient_colors(num_points: int, 
                           start_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                           end_color: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """
    生成渐变色数组
    
    Args:
        num_points: 点的数量
        start_color: 起始颜色 (R, G, B)
        end_color: 结束颜色 (R, G, B)
    
    Returns:
        colors: shape (num_points, 3) 的颜色数组
    """
    colors = np.zeros((num_points, 3), dtype=np.float32)
    t = np.linspace(0, 1, num_points)
    
    for i in range(3):
        colors[:, i] = start_color[i] * (1 - t) + end_color[i] * t
    
    return colors

class ReplayCloudDataRunner:
    def __init__(self,
                 transform_params: DictConfig,
                 env_params: DictConfig,
                 shape_meta: DictConfig,
                 tcp_ensemble_buffer_params: DictConfig,
                 gripper_ensemble_buffer_params: DictConfig,
                 output_dir: str = '',
                 use_relative_action: bool = False,
                 use_relative_tcp_obs_for_relative_action: bool = True,
                 action_interpolation_ratio: int = 1,
                 eval_episodes=10,
                 max_duration_time: float = 30,
                 tcp_action_update_interval: int = 6,
                 gripper_action_update_interval: int = 10,
                 tcp_pos_clip_range: ListConfig = ListConfig([[0.6, -0.4, 0.03], [1.0, 0.45, 0.4]]),
                 tcp_rot_clip_range: ListConfig = ListConfig([[-np.pi, 0., np.pi], [-np.pi, 0., np.pi]]),
                 tqdm_interval_sec = 5.0,
                 control_fps: float = 12,
                 inference_fps: float = 6,
                 latency_step: int = 0,
                 gripper_latency_step: Optional[int] = None,
                 n_obs_steps: int = 2,
                 obs_temporal_downsample_ratio: int = 2,
                 dataset_obs_temporal_downsample_ratio: int = 1,
                 downsample_extended_obs: bool = True,
                 enable_video_recording: bool = False,
                 vcamera_server_ip: Optional[Union[str, ListConfig]] = None,
                 vcamera_server_port: Optional[Union[int, ListConfig]] = None,
                 task_name=None,
                 debug: bool = True,
                 episode_to_replay: int = 0) -> None:
        
        # 初始化ROS和环境
        rclpy.init(args=None)
        self.transforms = RealWorldTransforms(option=transform_params)
        self.env = RealRobotEnvironment(
            transforms=self.transforms,
            robot_server_ip=env_params.get('robot_server_ip', 'localhost'),
            robot_server_port=env_params.get('robot_server_port', 50051),
            device_mapping_server_ip=env_params.get('device_mapping_server_ip', 'localhost'),
            device_mapping_server_port=env_params.get('device_mapping_server_port', 50052),
            data_processing_params=env_params.get('data_processing_params', {})
        )
        self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)

        # 加载数据集

        from diffusion_policy.dataset.cloud_pick_and_place_image_dataset import CloudPickAndPlaceImageDataset
        self.dataset = CloudPickAndPlaceImageDataset(
            identifier="arrange_mouse",
            debug=True
        )

        # 读取TCP姿态数据
        self.zarr_data = zarr.open(self.dataset.zarr_path + 'replay_buffer.zarr', mode='r')

        self.tcp_pose = np.array(self.zarr_data['data/left_robot_tcp_pose'])
        self.episode_ends = np.array(self.zarr_data['meta/episode_ends'])

        self.start_idx = self.episode_ends[episode_to_replay - 1] if episode_to_replay > 0 else 0
        self.end_idx = self.episode_ends[episode_to_replay]
        self.tcp_pose = self.tcp_pose[self.start_idx:self.end_idx]

        logger.info(f"Loaded {self.tcp_pose.shape[0]} TCP poses from the dataset.")

        # 初始化Rerun可视化
        rr.init("replay_cloud_data", spawn=True)

        # 转换为4x4齐次变换矩阵pick and place
        tcp_homo = pose_3d_9d_to_homo_matrix_batch(self.tcp_pose)
        
        # 生成渐变色
        self.colors = generate_gradient_colors(len(tcp_homo))

        # 可视化原始TCP姿态
        visualize_pose_matrices(
            tcp_homo,
            "tcp_original",
            colors=self.colors,
            show_coordinate_frame=False,
        )

        # 计算相对姿态
        self.tcp_relative = absolute_actions_to_relative_actions(self.tcp_pose)
        self.tcp_relative_homo = pose_3d_9d_to_homo_matrix_batch(self.tcp_relative)
        self.tcp_relative_homo = np.array(self.tcp_relative_homo)
        
        # 可视化相对姿态
        visualize_pose_matrices(self.tcp_relative_homo, "tcp_relative", colors=self.colors, show_coordinate_frame=False)

    @staticmethod
    def spin_executor(executor):
        executor.spin()

    def run(self):
        """运行回放和可视化"""
        executor = MultiThreadedExecutor()
        executor.add_node(self.env)

        spin_thread = threading.Thread(target=self.spin_executor, args=(executor,), daemon=True)
        spin_thread.start()
        time.sleep(2)

        # 开始回放
        self.env.reset()
        # 设置夹爪到最大宽度
        self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
        time.sleep(1)

        # 获取机器人初始姿态
        obs = self.env.get_obs()
        robot_tcp = obs['left_robot_tcp_pose']
        
        # 计算预期的机器人TCP姿态序列
        tcp_robot = relative_actions_to_absolute_actions(self.tcp_relative, base_absolute_action=robot_tcp[0])

        tcp_robot = np.array(tcp_robot)

        tcp_robot = pose_3d_9d_to_homo_matrix_batch(tcp_robot)

        # 可视化预期的机器人TCP轨迹
        visualize_pose_matrices(tcp_robot, "tcp_robot_expected", 
                              colors=self.colors, show_coordinate_frame=True)

        # 回放完整数据
        frame_rate = 10
        target_interval = 1.0 / frame_rate
        next_frame_time = time.time()
        
        actual_positions = []
        
        for i in range(0, len(tcp_robot)):
            tcp_6d = matrix4x4_to_pose_6d(tcp_robot[i])
            
            # 执行动作（如果需要的话）
            self.env.execute_action(np.concatenate([tcp_6d, np.zeros(10)]))
            
            # 获取当前实际姿态
            obs = self.env.get_obs()
            current_robot_tcp = obs['left_robot_tcp_pose'][0][:3]
            actual_positions.append(current_robot_tcp)
            
            # 可视化当前实际位置
            rr.log(f"tcp_robot_actual/frame_{i}", 
                   rr.Points3D([current_robot_tcp], colors=[[0, 1, 0]]))
            
            logger.debug(f"Replayed TCP robot pose {i}: position={tcp_robot[i][:3, 3]}")
            
            # 控制帧率
            time.sleep(max(0, next_frame_time - time.time()))
            next_frame_time += target_interval
        
        # 可视化完整的实际轨迹
        if actual_positions:
            actual_positions = np.array(actual_positions)
            green_colors = np.tile([0, 1, 0], (len(actual_positions), 1))
            rr.log("tcp_robot_actual_trajectory", 
                   rr.Points3D(actual_positions, colors=green_colors))
            
        logger.info("Replay completed successfully!")
            
