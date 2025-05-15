import threading
import time
import os.path as osp
import numpy as np
import torch
import tqdm
from loguru import logger
from typing import Dict, Tuple, Union, Optional, Any
import rclpy
import transforms3d as t3d
import py_cli_interaction
from rclpy.executors import MultiThreadedExecutor
from omegaconf import DictConfig, ListConfig
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.precise_sleep import precise_sleep
from diffusion_policy.common.ring_buffer import RingBuffer
from diffusion_policy.env.real_bimanual.real_env import RealRobotEnvironment
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict)
from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
from diffusion_policy.common.ensemble import EnsembleBuffer
from diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
)
import requests
import cv2
from copy import deepcopy


from gr00t.data.dataset import ModalityConfig
from gr00t.model.policy import BasePolicy
from gr00t.eval.service import BaseInferenceClient
    
class RobotInferenceClient(BaseInferenceClient, BasePolicy):
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)


# 设置CPU线程限制
import os
import psutil
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
cv2.setNumThreads(12)

# 设置CPU亲和性
total_cores = psutil.cpu_count()
num_cores_to_bind = 10
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
os.sched_setaffinity(0, cores_to_bind)

class RealPickAndPlaceGr00tRunner:
    def __init__(self,
                 gr00t_server_ip: str,
                 gr00t_server_port: int,
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
                 infer_res_temporal_downsample_ratio: int = 1,
                 debug: bool = True
                 ):
        self.gr00t_server_ip = gr00t_server_ip
        self.gr00t_server_port = gr00t_server_port
        if RobotInferenceClient is not None:
            self.gr00t_client = RobotInferenceClient(host=gr00t_server_ip, port=gr00t_server_port)
        else:
            self.gr00t_client = None
            logger.error("GR00T client not available. Please install gr00t package.")
        
        self.task_name = task_name
        self.transforms = RealWorldTransforms(option=transform_params)
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes
        self.debug = debug

        # 获取RGB和低维观察键
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        rclpy.init(args=None)
        self.env = RealRobotEnvironment(transforms=self.transforms, **env_params)

        self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
        time.sleep(2)

        self.max_duration_time = max_duration_time
        self.tcp_action_update_interval = tcp_action_update_interval
        self.gripper_action_update_interval = gripper_action_update_interval
        self.tcp_pos_clip_range = tcp_pos_clip_range
        self.tcp_rot_clip_range = tcp_rot_clip_range
        self.tqdm_interval_sec = tqdm_interval_sec
        self.control_fps = control_fps
        self.control_interval_time = 1.0 / control_fps
        self.inference_fps = inference_fps
        self.inference_interval_time = 1.0 / inference_fps
        assert self.control_fps % self.inference_fps == 0
        self.latency_step = latency_step
        self.gripper_latency_step = gripper_latency_step if gripper_latency_step is not None else latency_step
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.downsample_extended_obs = downsample_extended_obs
        self.tcp_ensemble_buffer = EnsembleBuffer(**tcp_ensemble_buffer_params)
        self.gripper_ensemble_buffer = EnsembleBuffer(**gripper_ensemble_buffer_params)
        self.use_relative_action = use_relative_action
        self.use_relative_tcp_obs_for_relative_action = use_relative_tcp_obs_for_relative_action
        self.action_interpolation_ratio = action_interpolation_ratio
        self.infer_res_temporal_downsample_ratio = infer_res_temporal_downsample_ratio

        # 视频录制设置
        self.enable_video_recording = enable_video_recording
        if enable_video_recording:
            assert isinstance(vcamera_server_ip, str) and isinstance(vcamera_server_port, int) or \
                     isinstance(vcamera_server_ip, ListConfig) and isinstance(vcamera_server_port, ListConfig), \
                "vcamera_server_ip and vcamera_server_port should be a string or ListConfig."
        if isinstance(vcamera_server_ip, str):
            vcamera_server_ip_list = [vcamera_server_ip]
            vcamera_server_port_list = [vcamera_server_port]
        elif isinstance(vcamera_server_ip, ListConfig):
            vcamera_server_ip_list = list(vcamera_server_ip)
            vcamera_server_port_list = list(vcamera_server_port)
        self.vcamera_server_ip_list = vcamera_server_ip_list
        self.vcamera_server_port_list = vcamera_server_port_list
        self.video_dir = osp.join(output_dir, 'videos')

        self.stop_event = threading.Event()
        self.session = requests.Session()

    @staticmethod
    def spin_executor(executor):
        executor.spin()

    def pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        # 复制观察字典以避免修改原始数据
        obs_dict = deepcopy(obs_dict)

        # 处理低维观察
        for key in self.lowdim_keys:
            if "wrt" not in key:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        # 计算抓手相对动作
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in self.lowdim_keys:
            obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        # 保存绝对观察数据
        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs_dict[key].copy()

        # 转换绝对动作为相对动作
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = obs_dict[key][-1].copy()
                    obs_dict[key] = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)

        return obs_dict, absolute_obs_dict

    def post_process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """处理动作数据，进行剪裁和格式转换"""
        assert len(action.shape) == 2  # (action_steps, d_a)
        
        # 处理6D旋转表示
        if self.env.data_processing_manager.use_6d_rotation:
            if action.shape[-1] == 4 or action.shape[-1] == 8:
                # 转换为6D姿态
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                # 使用默认欧拉角为0
                left_euler_batch = np.zeros_like(left_trans_batch)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 8:
                    right_trans_batch = action[:, 3:6]  # (action_steps, 3)
                    right_euler_batch = np.zeros_like(right_trans_batch)
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            elif action.shape[-1] == 10 or action.shape[-1] == 20:
                # 转换为6D姿态
                left_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 3:9])  # (action_steps, 3, 3)
                left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 20:
                    right_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 12:18])
                    right_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in right_rot_mat_batch])
                    right_trans_batch = action[:, 9:12]
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
        # 剪裁动作 (x, y, z)
        left_action_6d[:, :3] = np.clip(left_action_6d[:, :3], np.array(self.tcp_pos_clip_range[0]), np.array(self.tcp_pos_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, :3] = np.clip(right_action_6d[:, :3], np.array(self.tcp_pos_clip_range[2]), np.array(self.tcp_pos_clip_range[3]))
            
        # 剪裁动作 (r, p, y)
        left_action_6d[:, 3:] = np.clip(left_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[0]), np.array(self.tcp_rot_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, 3:] = np.clip(right_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[2]), np.array(self.tcp_rot_clip_range[3]))
            
        # 添加抓手动作
        if action.shape[-1] == 4:
            left_action = np.concatenate([left_action_6d, action[:, 3][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 8:
            left_action = np.concatenate([left_action_6d, action[:, 6][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 7][:, np.newaxis],
                                           np.zeros((action.shape[0], 1))], axis=1)
        elif action.shape[-1] == 10:
            left_action = np.concatenate([left_action_6d, action[:, 9][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 20:
            left_action = np.concatenate([left_action_6d, action[:, 18][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 19][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
        else:
            raise NotImplementedError

        if right_action is None:
            right_action = left_action.copy()
            is_bimanual = False
        else:
            is_bimanual = True
            
        action_all = np.concatenate([left_action, right_action], axis=-1)
        return (action_all, is_bimanual)

    def action_command_thread(self, stop_event):
        """动作执行线程，不断从集合缓冲区获取动作并发送到机器人"""
        while not stop_event.is_set():
            start_time = time.time()
            # 从集合缓冲区获取步骤动作
            tcp_step_action = self.tcp_ensemble_buffer.get_action()
            gripper_step_action = self.gripper_ensemble_buffer.get_action()
            
            if tcp_step_action is None or gripper_step_action is None:  # 缓冲区中没有动作 => 不移动
                cur_time = time.time()
                precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                logger.debug(f"Step: {self.action_step_count}, control_interval_time: {self.control_interval_time}, "
                             f"cur_time-start_time: {cur_time - start_time}")
                self.action_step_count += 1
                continue

            # 合并TCP和抓手动作
            combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)
            
            # 转换为16维机器人动作 (TCP + 抓手, 两臂)
            if self.debug:
                logger.debug(f"Step: {self.action_step_count}, combined_action: {combined_action[np.newaxis, :]}")
            step_action, is_bimanual = self.post_process_action(combined_action[np.newaxis, :])
            step_action = step_action.squeeze(0)

            # 发送动作到机器人
            if self.debug:
                logger.debug(f"Step: {self.action_step_count}, Send action to the robot: {step_action}")
            else:
                self.env.execute_action(step_action, use_relative_action=False, is_bimanual=is_bimanual)

            cur_time = time.time()
            precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
            self.action_step_count += 1

    def start_record_video(self, video_path):
        """启动视频录制"""
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/start_recording/{video_path}')
            if response.status_code == 200:
                logger.info(f"Start recording video to {video_path}")
            else:
                logger.error(f"Failed to start recording video to {video_path}")

    def stop_record_video(self):
        """停止视频录制"""
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/stop_recording')
            if response.status_code == 200:
                logger.info(f"Stop recording video")
            else:
                logger.error(f"Failed to stop recording video")

    def run(self, ):
        """运行主循环"""
        # 检查GR00T客户端是否可用
        if self.gr00t_client is None:
            logger.error("GR00T client not initialized. Aborting.")
            return
            
        # 获取GR00T模型的模态配置
        try:
            modality_configs = self.gr00t_client.get_modality_config()
            logger.info(f"Available GR00T modality configs: {list(modality_configs.keys())}")
        except Exception as e:
            logger.error(f"Failed to get GR00T modality config: {e}")
            return
            
        # 创建执行器并添加环境节点
        executor = MultiThreadedExecutor()
        executor.add_node(self.env)

        try:
            # 启动执行器线程
            spin_thread = threading.Thread(target=self.spin_executor, args=(executor,), daemon=True)
            spin_thread.start()

            time.sleep(2)
            for episode_idx in tqdm.tqdm(range(0, self.eval_episodes),
                                         desc=f"Eval in Real {self.task_name} with GR00T",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                logger.info(f"Start evaluation episode {episode_idx}")
                # 询问用户环境重置是否完成
                reset_flag = py_cli_interaction.parse_cli_bool('Has the environment reset finished?', default_value=True)
                if not reset_flag:
                    logger.warning("Skip this episode.")
                    continue

                logger.info("Start episode rollout.")
                # 开始回合
                self.env.reset()
                # 设置抓手到最大宽度
                self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
                time.sleep(1)

                # 清空集合缓冲区
                self.tcp_ensemble_buffer.clear()
                self.gripper_ensemble_buffer.clear()
                logger.debug("Reset environment and policy.")

                # 开始视频录制
                if self.enable_video_recording:
                    video_path = os.path.join(self.video_dir, f'episode_{episode_idx}.mp4')
                    self.start_record_video(video_path)
                    logger.info(f"Start recording video to {video_path}")

                # 准备启动动作线程
                self.stop_event.clear()
                time.sleep(0.5)
                # 启动动作命令线程
                action_thread = threading.Thread(target=self.action_command_thread, args=(self.stop_event,),
                                                 daemon=True)
                action_thread.start()

                # 初始化计数器和时间戳
                self.action_step_count = 0
                step_count = 0
                steps_per_inference = int(self.control_fps / self.inference_fps)
                start_timestamp = time.time()
                last_timestamp = start_timestamp
                
                try:
                    while True:
                        start_time = time.time()
                        # 获取观察
                        obs = self.env.get_obs(
                            obs_steps=self.n_obs_steps,
                            temporal_downsample_ratio=self.obs_temporal_downsample_ratio)

                        if len(obs) == 0:
                            logger.warning("No observation received! Skip this step.")
                            cur_time = time.time()
                            precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                            step_count += steps_per_inference
                            continue

                        # 创建观察字典
                        np_obs_dict = dict(obs)
                        # 获取变换后的真实观察字典
                        np_obs_dict = get_real_obs_dict(
                            env_obs=np_obs_dict, shape_meta=self.shape_meta)
                        np_obs_dict, np_absolute_obs_dict = self.pre_process_obs(np_obs_dict)
                        
                        # 转换图像数据格式
                        if 'left_wrist_img' in np_obs_dict:
                            np_obs_dict['left_wrist_img'] = (np_obs_dict['left_wrist_img'].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                        if 'external_img' in np_obs_dict:
                            np_obs_dict['external_img'] = (np_obs_dict['external_img'].transpose(0, 2, 3, 1) * 255).astype(np.uint8)

                        if self.debug:
                            logger.debug(f"Step: {step_count}, Get raw observation: {np_obs_dict}, image shape: {np_obs_dict['left_wrist_img'].shape}")

                        policy_time = time.time()
                        # 运行GR00T策略
                        
                        # 根据GR00T模态配置准备观察数据
                        gr00t_obs = {}
                        
                        # 添加视频数据
                        if 'video.wrist_view' in modality_configs:
                            # 调整图像大小为GR00T模型需要的尺寸(通常是256x256)
                            wrist_img = cv2.resize(np_obs_dict['left_wrist_img'][-1], (256, 256))
                            gr00t_obs['video.wrist_view'] = wrist_img[np.newaxis, ...]
                            
                        if 'video.external_view' in modality_configs:
                            ext_img = cv2.resize(np_obs_dict['external_img'][-1], (256, 256))
                            gr00t_obs['video.external_view'] = ext_img[np.newaxis, ...]
                            
                        # 合并状态数据为单个10维向量：位置(3)+旋转(6)+抓手宽度(1)
                        if 'state.tcp_position' in modality_configs or 'state.tcp_rotation' in modality_configs or 'state.gripper_width' in modality_configs:
                            # 创建状态向量
                            tcp_pose = np_absolute_obs_dict['left_robot_tcp_pose'][-1]
                            gripper_width = np_absolute_obs_dict['left_robot_gripper_width'][-1]
                            
                            # 分解状态数据
                            gr00t_obs['state.tcp_position'] = tcp_pose[:3][np.newaxis, ...]  # 位置 (0-3)
                            gr00t_obs['state.tcp_rotation'] = tcp_pose[3:9][np.newaxis, ...]  # 旋转 (3-9)
                            gr00t_obs['state.gripper_width'] = gripper_width[np.newaxis, ...]  # 抓手宽度 (9-10)
                            
                        # 添加任务描述
                        if 'annotation.human.action.task_description' in modality_configs:
                            gr00t_obs['annotation.human.action.task_description'] = ["Grasp and place an object."]
                            
                        # 添加有效性标记（如果需要）
                        if 'annotation.human.validity' in modality_configs:
                            gr00t_obs['annotation.human.validity'] = [1.0]  # 假设所有数据都有效
                            
                        # 调用GR00T模型获取动作
                        try:
                            gr00t_infer_res = self.gr00t_client.get_action(gr00t_obs)
                            logger.debug(f"GR00T inference time: {time.time() - policy_time:.3f}s")
                            
                            # 转换GR00T输出动作到环境所需格式
                            action_all = np.zeros((16, 10))  # 16步，每步10维动作
                            
                            # 填充位置动作
                            if 'action.tcp_position' in gr00t_infer_res:
                                action_all[:, :3] = gr00t_infer_res['action.tcp_position']
                            
                            # 填充旋转动作
                            if 'action.tcp_rotation' in gr00t_infer_res:
                                action_all[:, 3:9] = gr00t_infer_res['action.tcp_rotation']
                            
                            # 填充抓手动作
                            if 'action.gripper_width' in gr00t_infer_res:
                                action_all[:, 9] = gr00t_infer_res['action.gripper_width'].squeeze(-1)
                                
                            logger.debug(f"Step: {step_count}, Get GR00T action: {action_all}")
                            
                        except Exception as e:
                            logger.error(f"GR00T inference failed: {e}")
                            cur_time = time.time()
                            precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                            step_count += steps_per_inference
                            continue

                        if self.use_relative_action:
                            base_absolute_action = np.concatenate([
                                np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                                np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                            ], axis=-1)
                            action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)

                        if self.action_interpolation_ratio > 1:
                            action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)

                        # 更新TCP动作
                        if step_count % self.tcp_action_update_interval == 0:
                            if action_all.shape[-1] == 4:
                                tcp_action = action_all[self.latency_step:, :3]
                            elif action_all.shape[-1] == 8:
                                tcp_action = action_all[self.latency_step:, :6]
                            elif action_all.shape[-1] == 10:
                                tcp_action = action_all[self.latency_step:, :9]
                            elif action_all.shape[-1] == 20:
                                tcp_action = action_all[self.latency_step:, :18]
                            else:
                                raise NotImplementedError
                            tcp_action = tcp_action[::self.infer_res_temporal_downsample_ratio]
                            # 添加到集合缓冲区
                            logger.debug(f"Step: {step_count}, Add TCP action to ensemble buffer: {tcp_action}")
                            self.tcp_ensemble_buffer.add_action(tcp_action, step_count)

                            if self.env.enable_exp_recording:
                                self.env.get_predicted_action(tcp_action, type='full_tcp')

                        # 更新抓手动作
                        if step_count % self.gripper_action_update_interval == 0:
                            if action_all.shape[-1] == 4:
                                gripper_action = action_all[self.gripper_latency_step:, 3:]
                            elif action_all.shape[-1] == 8:
                                gripper_action = action_all[self.gripper_latency_step:, 6:]
                            elif action_all.shape[-1] == 10:
                                gripper_action = action_all[self.gripper_latency_step:, 9:]
                            elif action_all.shape[-1] == 20:
                                gripper_action = action_all[self.gripper_latency_step:, 18:]
                            else:
                                raise NotImplementedError
                            gripper_action = gripper_action[::(self.infer_res_temporal_downsample_ratio + 1)]
                            # 添加到集合缓冲区
                            logger.debug(f"Step: {step_count}, Add gripper action to ensemble buffer: {gripper_action}")
                            self.gripper_ensemble_buffer.add_action(gripper_action, step_count)

                            if self.env.enable_exp_recording:
                                self.env.get_predicted_action(gripper_action, type='full_gripper')

                        cur_time = time.time()
                        precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                        if cur_time - start_timestamp >= self.max_duration_time:
                            logger.info(f"Episode {episode_idx} reaches max duration time {self.max_duration_time} seconds.")
                            break
                        step_count += steps_per_inference

                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt! Terminate the episode now!")
                except Exception as e:
                    logger.error(f"Error during episode execution: {e}")
                finally:
                    self.stop_event.set()
                    action_thread.join()
                    if self.enable_video_recording:
                        self.stop_record_video()
                    self.env.save_exp(episode_idx)

            # 支持成功计数
            spin_thread.join()
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
        finally:
            self.env.destroy_node()