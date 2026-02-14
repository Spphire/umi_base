import cv2
import numpy as np
import open3d as o3d
from loguru import logger
from typing import Dict, Optional
from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from diffusion_policy.common.data_models import SensorMessage, SensorMode
from diffusion_policy.common.visualization_utils import visualize_pcd_from_numpy, visualize_rgb_image
from diffusion_policy.common.space_utils import pose_6d_to_4x4matrix, pose_6d_to_pose_9d, pose_7d_to_pose_6d
from diffusion_policy.common.image_utils import center_crop_and_resize_image
from omegaconf import DictConfig
import bson
import os
import json

from post_process_scripts.head_data_process_util import get_full_data
from post_process_scripts.ArUco_calibration import run_aruco_world_pnp_verbose, poses_to_T, T_to_pose, unity2zup_right_frame_batch
from scipy.spatial.transform import Rotation as R

class DataPostProcessingManager:
    def __init__(self,
                 transforms: RealWorldTransforms,
                 mode: str = 'single_arm_one_realsense',
                 image_resize_shape: tuple = (320, 240),
                 # [todo] support 'center_crop', 'cv2', 'center_pad'
                 image_resize_mode: str = 'cv2',
                 use_6d_rotation: bool = True,
                 debug: bool = True):
        self.transforms = transforms
        self.mode = SensorMode[mode]
        self.use_6d_rotation = use_6d_rotation
        self.resize_shape = image_resize_shape
        self.resize_mode = image_resize_mode
        self.debug = debug

    def resize_img(self, img: np.ndarray) -> np.ndarray:
        if self.resize_mode == 'cv2':
            return cv2.resize(img, dsize=self.resize_shape)
        if self.resize_mode == 'center_crop':
            return center_crop_and_resize_image(img, self.resize_shape)
        raise NotImplementedError

    def convert_sensor_msg_to_obs_dict(self, sensor_msg: SensorMessage) -> Dict[str, np.ndarray]:
        obs_dict = dict()
        obs_dict['timestamp'] = np.array([sensor_msg.timestamp])
        # logger.info(f"left_robot_tcp_pose: {sensor_msg.leftRobotTCP}")
        # logger.info(
        #     f"left_robot_gripper_width: {sensor_msg.leftRobotGripperState[0][np.newaxis]}"
        # )
        # Add independent key-value pairs for left robot
        obs_dict['left_robot_tcp_pose'] = sensor_msg.leftRobotTCP
        obs_dict['left_robot_tcp_vel'] = sensor_msg.leftRobotTCPVel
        obs_dict['left_robot_tcp_wrench'] = sensor_msg.leftRobotTCPWrench
        obs_dict['left_robot_gripper_width'] = sensor_msg.leftRobotGripperState[0][np.newaxis]
        obs_dict['left_robot_gripper_force'] = sensor_msg.leftRobotGripperState[1][np.newaxis]

        # Add independent key-value pairs for right robot
        obs_dict['right_robot_tcp_pose'] = sensor_msg.rightRobotTCP
        obs_dict['right_robot_tcp_vel'] = sensor_msg.rightRobotTCPVel
        obs_dict['right_robot_tcp_wrench'] = sensor_msg.rightRobotTCPWrench
        obs_dict['right_robot_gripper_width'] = sensor_msg.rightRobotGripperState[0][np.newaxis]
        obs_dict['right_robot_gripper_force'] = sensor_msg.rightRobotGripperState[1][np.newaxis]

        if self.use_6d_rotation:
            obs_dict['left_robot_tcp_pose'] = pose_6d_to_pose_9d(sensor_msg.leftRobotTCP)
            obs_dict['right_robot_tcp_pose'] = pose_6d_to_pose_9d(sensor_msg.rightRobotTCP)

        if self.debug:
            logger.debug(f'left_robot_tcp_pose: {obs_dict["left_robot_tcp_pose"]}, '
                         f'right_robot_tcp_pose: {obs_dict["right_robot_tcp_pose"]}')
            logger.debug(f'left_robot_tcp_vel: {obs_dict["left_robot_tcp_vel"]}, '
                            f'right_robot_tcp_vel: {obs_dict["right_robot_tcp_vel"]}')
            logger.debug(f'left_robot_tcp_wrench: {obs_dict["left_robot_tcp_wrench"]}, '
                            f'right_robot_tcp_wrench: {obs_dict["right_robot_tcp_wrench"]}')
            logger.debug(f'left_robot_gripper_width: {obs_dict["left_robot_gripper_width"]}, '
                            f'right_robot_gripper_width: {obs_dict["right_robot_gripper_width"]}')
            logger.debug(f'left_robot_gripper_force: {obs_dict["left_robot_gripper_force"]}, '
                            f'right_robot_gripper_force: {obs_dict["right_robot_gripper_force"]}')

        # TODO: make all sensor post-processing in parallel
        # logger.info(
        #     f"left_robot_wrist_img shape: {sensor_msg.leftWristCameraRGB.shape}"
        # )
        obs_dict['left_wrist_img'] = self.resize_img(sensor_msg.leftWristCameraRGB)

        if self.debug:
            visualize_rgb_image(obs_dict['left_wrist_img'])

        obs_dict['external_img'] = self.resize_img(sensor_msg.externalCameraRGB)
        if self.debug:
            visualize_rgb_image(obs_dict['external_img'])
        if self.mode == SensorMode.single_arm_two_realsense or self.mode == SensorMode.single_arm_one_realsense:
            return obs_dict

        obs_dict['right_wrist_img'] = self.resize_img(sensor_msg.rightWristCameraRGB)
        obs_dict['right_gripper1_img'] = self.resize_img(sensor_msg.rightGripperCameraRGB1)
        obs_dict['right_gripper2_img'] = self.resize_img(sensor_msg.rightGripperCameraRGB2)


        if self.debug:
            visualize_rgb_image(obs_dict['right_wrist_img'])
            visualize_rgb_image(obs_dict['right_gripper1_img'])
            visualize_rgb_image(obs_dict['right_gripper2_img'])

            return obs_dict
        else:
            raise NotImplementedError


class DataPostProcessingManageriPhone:
    def __init__(self,
                 image_resize_shape: tuple = (320, 240),
                 use_6d_rotation: bool = True,
                 debug: bool = False):
        self.use_6d_rotation = use_6d_rotation
        self.resize_shape = image_resize_shape
        self.debug = debug

    @staticmethod
    def load_bson_file(file_path):
        try:
            with open(file_path, 'rb') as f:
                bson_data = f.read()
            try:
                bson_dict = bson.loads(bson_data) # bson library
            except AttributeError as e:
                bson_dict = bson.decode(bson_data) # pymongo library
            
            return bson_dict
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def get_numpy_arrays(data):
        timestamps = np.array(data.get('timestamps', []))
        arkit_poses = np.array(data.get('arkitPose', []))
        gripper_widths = np.array(data.get('gripperWidth', []))

        return timestamps, arkit_poses, gripper_widths

    def read_bson(self, file_path):
        data = self.load_bson_file(file_path)
        if data is None:
            return

        timestamps, arkit_poses, gripper_widths = self.get_numpy_arrays(data)
        return timestamps, arkit_poses, gripper_widths

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        cap.release()
        frames_array = np.array(frames)
        return frames_array

    def extract_msg_to_obs_dict(
        self,
        session: Dict,
        clip_head_seconds: float = 0.0,
        clip_tail_seconds: float = 0.0
    ) -> Optional[Dict[str, np.ndarray]]:
        obs_dict = dict()

        timestamps = {}
        arkit_poses = {}
        gripper_widths = {}

        for record in session.values():
            metadata_path = os.path.join(record, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            camera_position = metadata['camera_position'] if 'camera_position' in metadata else 'left_wrist'

            bson_path = os.path.join(record, "frame_data.bson")
            t, a, g = self.read_bson(bson_path)
            
            if t is None or len(t) == 0:
                continue

            timestamps[camera_position] = t
            arkit_poses[camera_position] = a
            gripper_widths[camera_position] = g

        if not timestamps:
            return None

        latest_start_time = max([t[0] for t in timestamps.values()])
        earliest_end_time = min([t[-1] for t in timestamps.values()])
        
        clipped_start_time = latest_start_time + clip_head_seconds
        clipped_end_time = earliest_end_time - clip_tail_seconds
        
        if clipped_start_time >= clipped_end_time:
            return None

        start_frame_indices = {}
        end_frame_indices = {}
        for k, v in timestamps.items():
            start_frame_indices[k] = np.searchsorted(v, clipped_start_time, side='left')
            end_frame_indices[k] = np.searchsorted(v, clipped_end_time, side='right')

        num_frames = min([end_frame_indices[k] - start_frame_indices[k] for k in timestamps.keys()])
        
        if num_frames <= 0:
            return None

        # Project all data to start_frame - start_frame + num_frames
        def project_data(data_dict, start_indices, num_frames):
            projected_data = {}
            for k, v in data_dict.items():
                start_idx = start_indices[k]
                projected_data[k] = v[start_idx:start_idx + num_frames]
            return projected_data
        
        timestamps, arkit_poses, gripper_widths = \
            project_data(timestamps, start_frame_indices, num_frames), \
            project_data(arkit_poses, start_frame_indices, num_frames), \
            project_data(gripper_widths, start_frame_indices, num_frames)

        # convert quat format from [x, y, z, w] to [w, x, y, z]
        for k in arkit_poses.keys():
            arkit_poses[k][:, 3:] = arkit_poses[k][:, [6,3,4,5]]

        # extend dims
        for k in timestamps.keys():
            timestamps[k] = timestamps[k][:, np.newaxis]
            gripper_widths[k] = gripper_widths[k][:, np.newaxis]

        if 'left_wrist' not in timestamps and 'right_wrist' not in timestamps:
            return None

        obs_dict['timestamp'] = timestamps['left_wrist'] if 'left_wrist' in timestamps else timestamps['right_wrist']

        for k in arkit_poses.keys():
            key_prefix = "right" if "right" in k else "left"
            if self.use_6d_rotation:
                a_9d = [pose_6d_to_pose_9d(pose_7d_to_pose_6d(pose)) for pose in arkit_poses[k]]
                obs_dict[f'{key_prefix}_robot_tcp_pose'] = a_9d
            else:
                obs_dict[f'{key_prefix}_robot_tcp_pose'] = arkit_poses[k]

            obs_dict[f'{key_prefix}_robot_gripper_width'] = gripper_widths[k]

            images_array = self.load_video_frames(os.path.join(session[k], "recording.mp4"))
            obs_dict[f'{key_prefix}_wrist_img'] = []
            for image in images_array[start_frame_indices[k]:start_frame_indices[k]+num_frames]:
                obs_dict[f'{key_prefix}_wrist_img'].append(cv2.resize(image, self.resize_shape))
            if self.debug:
                visualize_rgb_image(obs_dict[f'{key_prefix}_wrist_img'])

        return obs_dict


class DataPostProcessingManagerVR:
    def __init__(self,
                 image_resize_shape: tuple = (320, 240),
                 use_6d_rotation: bool = True,
                 debug: bool = False):
        self.use_6d_rotation = use_6d_rotation
        self.resize_shape = image_resize_shape
        self.debug = debug

    @staticmethod
    def load_bson_file(file_path):
        try:
            with open(file_path, 'rb') as f:
                bson_data = f.read()
            try:
                bson_dict = bson.loads(bson_data) # bson library
            except AttributeError as e:
                bson_dict = bson.decode(bson_data) # pymongo library

            return bson_dict
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def get_numpy_arrays(data):
        timestamps = np.array(data.get('timestamps', []))
        arkit_poses = np.array(data.get('arkitPose', []))
        gripper_widths = np.array(data.get('gripperWidth', []))

        return timestamps, arkit_poses, gripper_widths

    def read_bson(self, file_path):
        data = self.load_bson_file(file_path)
        if data is None:
            return None, None, None

        timestamps, arkit_poses, gripper_widths = self.get_numpy_arrays(data)
        return timestamps, arkit_poses, gripper_widths

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        cap.release()
        frames_array = np.array(frames)
        return frames_array

    def extract_msg_to_obs_dict(
        self,
        session: Dict,
        clip_head_seconds: float = 0.0,
        clip_tail_seconds: float = 0.0,
        use_aruco_calibration: bool = True
    ) -> Optional[Dict[str, np.ndarray]]:
        obs_dict = dict()

        timestamps = {}
        arkit_poses = {}
        gripper_widths = {}
        timestamps_proj = None
        
        head_data = None
        aruco_result = None

        for record in session.values():
            metadata_path = os.path.join(record, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            camera_position = metadata.get('camera_position', '')

            # 处理头部数据
            if camera_position in ['', 'head']:
                head_data = get_full_data(record)
                if use_aruco_calibration:
                    aruco_result = run_aruco_world_pnp_verbose(head_data)
                continue

            # 处理手臂数据
            bson_path = os.path.join(record, "frame_data.bson")
            t, a, g = self.read_bson(bson_path)
            if t is None or len(t) == 0:
                continue
            timestamps[camera_position] = t
            arkit_poses[camera_position] = a
            gripper_widths[camera_position] = g

        if head_data is None:
            return None

        if not timestamps:
            return None
        
        # 确定时间范围
        all_timestamps = []
        for k, v in timestamps.items():
            all_timestamps.extend([v[0], v[-1]])
        all_timestamps.extend([head_data['leftCameraAccessTimestamps'][0],head_data['leftCameraAccessTimestamps'][-1]])
        all_timestamps.extend([head_data['rightCameraAccessTimestamps'][0],head_data['rightCameraAccessTimestamps'][-1]])

        if not all_timestamps:
            return None

        latest_start_time = max(all_timestamps[::2])  # 所有开始时间中的最大值
        earliest_end_time = min(all_timestamps[1::2])  # 所有结束时间中的最小值
        ts_clip = (latest_start_time, earliest_end_time)


#=============================================
 # 匹配aruco的iphone帧（左手）
        def match_aruco_to_iphone(aruco_ts, iphone_ts):
            """
            对每个aruco timestamp找到最近的iphone timestamp
            自动剔除在iphone_ts范围之外的aruco_ts
            返回：
                filtered_aruco_ts: 过滤后的aruco_ts
                iphone_ts_correspond: 与filtered_aruco_ts对应的最近iphone_ts
            """
            aruco_ts = np.asarray(aruco_ts)
            iphone_ts = np.asarray(iphone_ts)

            # 1️⃣ 过滤越界的aruco_ts
            mask = (aruco_ts >= ts_clip[0]) & (aruco_ts <= ts_clip[1])
            filtered_aruco_ts = aruco_ts[mask]
            aruco_idx = np.nonzero(mask)[0]

            if len(filtered_aruco_ts) == 0:
                return np.array([]), np.array([])

            # 2️⃣ 搜索插入点（保证递增）
            idx = np.searchsorted(iphone_ts, filtered_aruco_ts, side='left')

            # 3️⃣ 边界处理
            idx0 = np.clip(idx - 1, 0, len(iphone_ts) - 1)
            idx1 = np.clip(idx, 0, len(iphone_ts) - 1)

            # 4️⃣ 取两侧最近
            diff0 = np.abs(filtered_aruco_ts - iphone_ts[idx0])
            diff1 = np.abs(filtered_aruco_ts - iphone_ts[idx1])
            iphone_idx = np.where(diff0 <= diff1, idx0, idx1)

            return aruco_idx, iphone_idx

        def get_unity2ar(chosen_aruco_id, chosen_arkit_label):

            frame_idx_0 = aruco_result['OK']['frame_idx'][chosen_aruco_id]
            aruco_timestamps = np.array(head_data['rightCameraAccessTimestamps'])[frame_idx_0]
            iphone_timestamps = timestamps[chosen_arkit_label].reshape(-1)

            aruco_match_idx, iphone_match_idx = match_aruco_to_iphone(aruco_timestamps, iphone_timestamps)

            aruco_iphone_poses_matched = aruco_result['OK']['iphone_camera_poses'][chosen_aruco_id][aruco_match_idx]
            arkit_poses_wxyz = arkit_poses[chosen_arkit_label][iphone_match_idx]
            arkit_poses_wxyz[:, 3:] = arkit_poses_wxyz[:, [6,3,4,5]]  # xyzw -> wxyz
            ar_iphone_poses_matched = poses_to_T(arkit_poses_wxyz)
            print(f"found {aruco_iphone_poses_matched.shape[0]} sample for aruco calibration")
            
            transform_matrix = ar_iphone_poses_matched @ np.linalg.inv(aruco_iphone_poses_matched)
            # ====median clip====
            t = transform_matrix[:, :3, 3]
            t_center = np.median(t, axis=0)

            d = np.linalg.norm(t - t_center, axis=1)
            idx1 = np.argsort(d)[:int(0.5 * len(d))]
            T_stage1 = transform_matrix[idx1]
            # ====mean clip====
            t1 = T_stage1[:, :3, 3]
            t_center2 = np.mean(t1, axis=0)

            d2 = np.linalg.norm(t1 - t_center2, axis=1)
            idx2 = np.argsort(d2)[:max(min(15, len(d2)), int(0.5 * len(d2)))]
            T_final = T_stage1[idx2]


            # ================= 平移散布 =================
            var_per_axis = np.sqrt(np.var(T_final[:,:3,3], axis=0, ddof=0))
            print(f"aruco{chosen_aruco_id} calibration var_per_axis: {var_per_axis}m")
            # ================= 旋转散布 =================
            rot_mats = T_final[:, :3, :3]
            # 旋转矩阵 → 旋转向量 (rad)
            rot_vecs = np.array([
                cv2.Rodrigues(R)[0].flatten() for R in rot_mats
            ])  # shape (N,3)
            # 每轴旋转标准差（RMS）
            rot_std_per_axis = np.sqrt(
                np.var(rot_vecs, axis=0, ddof=0)
            )
            print(f"aruco{chosen_aruco_id} calibration rotation std per axis: {rot_std_per_axis} rad")
            print(f"aruco{chosen_aruco_id} calibration rotation std per axis: {np.degrees(rot_std_per_axis)} deg")

            def average_transform(transform_matrix: np.ndarray) -> np.ndarray:
                """
                平均 N 个 4x4 变换矩阵
                transform_matrix: shape (N, 4, 4)
                返回: 平均 4x4 变换矩阵
                """
                N = transform_matrix.shape[0]
                
                # 1️⃣ 平移部分平均
                avg_t = np.mean(transform_matrix[:, :3, 3], axis=0)  # shape (3,)
                
                # 2️⃣ 旋转部分平均
                rot_mats = transform_matrix[:, :3, :3]  # (N,3,3)
                
                # 转四元数
                quats = np.array([R.from_matrix(Ri).as_quat() for Ri in rot_mats])  # shape (N,4) x,y,z,w
                # 四元数平均（按简单线性法，归一化）
                avg_quat = np.mean(quats, axis=0)
                avg_quat /= np.linalg.norm(avg_quat)
                
                # 转回旋转矩阵
                avg_R = R.from_quat(avg_quat).as_matrix()
                
                # 3️⃣ 构建平均变换矩阵
                avg_T = np.eye(4)
                avg_T[:3, :3] = avg_R
                avg_T[:3, 3] = avg_t
                
                return avg_T
            unity2ar = average_transform(T_final)
            return unity2ar
        
        unity2leftar, unity2rightar = None, None
        if use_aruco_calibration:
            try:
                chosen_aruco_id = 0
                chosen_arkit_label = 'left_wrist'
                unity2leftar = get_unity2ar(chosen_aruco_id, chosen_arkit_label)
            except Exception as e:
                print(e)
                print("failed to calculate unity2leftar transforms based on aruco calibration")
                return None
            
            if 'right_wrist' in timestamps.keys():
                try:
                    chosen_aruco_id = 1
                    chosen_arkit_label = 'right_wrist' 
                    unity2rightar = get_unity2ar(chosen_aruco_id, chosen_arkit_label)
                except Exception as e:
                    print(e)
                    print("failed to calculate unity2rightar transforms based on aruco calibration")
                    return None
            print("finished calculating unity2ar transforms based on aruco calibration")
        else:
            # 不使用ArUco校准时，使用单位矩阵（即不进行坐标转换）
            unity2leftar = np.eye(4)
            unity2rightar = np.eye(4)
            print("ArUco calibration disabled, using identity transform")





        clipped_start_time = latest_start_time + clip_head_seconds
        clipped_end_time = earliest_end_time - clip_tail_seconds

        if clipped_start_time >= clipped_end_time:
            return None

        # 计算帧索引
        start_frame_indices = {}
        end_frame_indices = {}
        for k, v in timestamps.items():
            start_frame_indices[k] = np.searchsorted(v, clipped_start_time, side='left')
            end_frame_indices[k] = np.searchsorted(v, clipped_end_time, side='right')

        # 计算共同的帧数
        num_frames = min([end_frame_indices[k] - start_frame_indices[k] for k in timestamps.keys()])
        if num_frames <= 0:
            return None
        
        # Project all data to start_frame - start_frame + num_frames
        def project_data(data_dict, start_indices, num_frames):
            projected_data = {}
            for k, v in data_dict.items():
                start_idx = start_indices[k]
                projected_data[k] = v[start_idx:start_idx + num_frames]
            return projected_data

        # 处理手臂数据
        if arkit_poses:
            arkit_poses = project_data(arkit_poses, start_frame_indices, num_frames)
            gripper_widths = project_data(gripper_widths, start_frame_indices, num_frames)
            timestamps_proj = project_data({k: v for k, v in timestamps.items() if k != 'head'},
                                         start_frame_indices, num_frames)

            # convert quat format from [x, y, z, w] to [w, x, y, z]
            for k in arkit_poses.keys():
                arkit_poses[k][:, 3:] = arkit_poses[k][:, [6,3,4,5]]

            # extend dims
            for k in timestamps_proj.keys():
                timestamps_proj[k] = timestamps_proj[k][:, np.newaxis]
                gripper_widths[k] = gripper_widths[k][:, np.newaxis]
                
        # 设置时间戳 以手臂为准（头部帧率高）
        if timestamps_proj:
            obs_dict['timestamp'] = list(timestamps_proj.values())[0]
      
        # 处理手臂数据
        for k in arkit_poses.keys():
            key_prefix = "right" if "right" in k else "left"
            if self.use_6d_rotation:
                a_9d = [pose_6d_to_pose_9d(pose_7d_to_pose_6d(pose)) for pose in arkit_poses[k]]
                obs_dict[f'{key_prefix}_robot_tcp_pose'] = a_9d
            else:
                obs_dict[f'{key_prefix}_robot_tcp_pose'] = arkit_poses[k]

            obs_dict[f'{key_prefix}_robot_gripper_width'] = gripper_widths[k]

            images_array = self.load_video_frames(os.path.join(session[k], "recording.mp4"))
            if images_array is not None:
                obs_dict[f'{key_prefix}_wrist_img'] = []
                for image in images_array[start_frame_indices[k]:start_frame_indices[k]+num_frames]:
                    obs_dict[f'{key_prefix}_wrist_img'].append(cv2.resize(image, self.resize_shape))
                if self.debug:
                    visualize_rgb_image(obs_dict[f'{key_prefix}_wrist_img'][0])

        # 匹配iphone的unity帧
        unity_ts = np.array(head_data['leftCameraAccessTimestamps'])
        end_frame_indices = np.searchsorted(unity_ts, clipped_end_time, side='right')
        _, unity_match_idx = match_aruco_to_iphone(timestamps_proj['left_wrist'].reshape(-1), unity_ts[:end_frame_indices])

        Rx_180 = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ]) # spetial for unity

        if use_aruco_calibration and aruco_result is not None:
            left_eye_poses = T_to_pose(unity2leftar @ aruco_result["cam_l_poses"][unity_match_idx] @ Rx_180)
            right_eye_poses = T_to_pose(unity2leftar @ aruco_result["cam_r_poses"][unity_match_idx] @ Rx_180)
        else:
            # 不使用ArUco校准时，直接使用Unity相机姿态（ARKit坐标系）
            left_cam_poses = np.array(head_data.get('leftCameraPoses', []))
            right_cam_poses = np.array(head_data.get('rightCameraPoses', []))
            
            if len(left_cam_poses) == 0 or len(right_cam_poses) == 0:
                logger.warning("Camera poses not found in head_data")
                return None
            print(left_cam_poses.shape)
            # 转换为变换矩阵并应用旋转
            left_eye_poses = T_to_pose(unity2zup_right_frame_batch(left_cam_poses[unity_match_idx]))
            right_eye_poses = T_to_pose(unity2zup_right_frame_batch(right_cam_poses[unity_match_idx]))

        if self.use_6d_rotation:
            a_9d = [pose_6d_to_pose_9d(pose_7d_to_pose_6d(pose)) for pose in left_eye_poses]
            obs_dict[f'left_eye_tcp_pose'] = a_9d

            a_9d = [pose_6d_to_pose_9d(pose_7d_to_pose_6d(pose)) for pose in right_eye_poses]
            obs_dict[f'right_eye_tcp_pose'] = a_9d
        else:
            obs_dict[f'left_eye_tcp_pose'] = left_eye_poses
            obs_dict[f'right_eye_tcp_pose'] = right_eye_poses

        obs_dict[f'left_eye_img'] = []
        for image in head_data["leftCameraFrames"][unity_match_idx]:
            obs_dict[f'left_eye_img'].append(cv2.resize(image, self.resize_shape))
        obs_dict[f'right_eye_img'] = []
        for image in head_data["rightCameraFrames"][unity_match_idx]:
            obs_dict[f'right_eye_img'].append(cv2.resize(image, self.resize_shape))
    
        if unity2rightar is not None and 'right_wrist' in arkit_poses:
            transformed_pose = T_to_pose(unity2leftar @ np.linalg.inv(unity2rightar) @ poses_to_T(np.array(arkit_poses['right_wrist'])))
            if self.use_6d_rotation:
                a_9d = [pose_6d_to_pose_9d(pose_7d_to_pose_6d(pose)) for pose in transformed_pose]
                obs_dict[f'right_robot_tcp_pose'] = a_9d
            else:
                obs_dict[f'right_robot_tcp_pose'] = transformed_pose

        # import pickle

        # with open("/mnt/data/shenyibo/workspace/umi_base/obs_dict.pkl", "wb") as f:
        #     pickle.dump(obs_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


        return obs_dict
