import cv2
import numpy as np
import open3d as o3d
from loguru import logger
from typing import Dict
from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from diffusion_policy.common.data_models import SensorMessage, SensorMode
from diffusion_policy.common.visualization_utils import visualize_pcd_from_numpy, visualize_rgb_image
from diffusion_policy.common.space_utils import pose_6d_to_4x4matrix, pose_6d_to_pose_9d, pose_7d_to_pose_6d
from diffusion_policy.common.image_utils import center_crop_and_resize_image
from omegaconf import DictConfig
import bson
import os
import json

class DataPostProcessingManager:
    def __init__(self,
                 transforms: RealWorldTransforms,
                 mode: str = 'single_arm_one_realsense',
                 image_resize_shape: tuple = (320, 240),
                 # [todo] support 'center_crop', 'cv2', 'center_pad'
                 image_resize_mode: str = 'cv2',
                 use_6d_rotation: bool = True,
                 debug: bool = False):
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

    def extract_msg_to_obs_dict(self, session: Dict) -> Dict[str, np.ndarray]:
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

            timestamps[camera_position] = t
            arkit_poses[camera_position] = a
            gripper_widths[camera_position] = g

        latest_start_time = max([t[0] for t in timestamps.values()])
        earliest_end_time = min([t[-1] for t in timestamps.values()])

        start_frame_indices = {}
        end_frame_indices = {}
        for k, v in timestamps.items():
            start_frame_indices[k] = np.searchsorted(v, latest_start_time, side='right')
            end_frame_indices[k] = np.searchsorted(v, earliest_end_time, side='left')

        num_frames = min([end_frame_indices[k] - start_frame_indices[k] for k in timestamps.keys()])

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

        obs_dict['timestamp'] = timestamps['left_wrist']

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
