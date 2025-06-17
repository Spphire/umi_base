import cv2
import numpy as np
import open3d as o3d
from loguru import logger
from typing import Dict
from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from diffusion_policy.common.data_models import SensorMessage, SensorMode
from diffusion_policy.common.visualization_utils import visualize_pcd_from_numpy, visualize_rgb_image
from diffusion_policy.common.space_utils import pose_6d_to_4x4matrix, pose_6d_to_pose_9d, pose_7d_to_pose_6d
from omegaconf import DictConfig
from bson import BSON
import os

class DataPostProcessingManager:
    def __init__(self,
                 transforms: RealWorldTransforms,
                 mode: str = 'single_arm_one_realsense',
                 image_resize_shape: tuple = (320, 240),
                 use_6d_rotation: bool = True,
                 debug: bool = False):
        self.transforms = transforms
        self.mode = SensorMode[mode]
        self.use_6d_rotation = use_6d_rotation
        self.resize_shape = image_resize_shape
        self.debug = debug

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
        obs_dict['left_wrist_img'] = cv2.resize(sensor_msg.leftWristCameraRGB, size=self.resize_shape)
        if self.debug:
            visualize_rgb_image(obs_dict['left_wrist_img'])

        obs_dict['external_img'] = cv2.resize(sensor_msg.externalCameraRGB, size=self.resize_shape)
        if self.debug:
            visualize_rgb_image(obs_dict['external_img'])
        if self.mode == SensorMode.single_arm_two_realsense or self.mode == SensorMode.single_arm_one_realsense:
            return obs_dict

        obs_dict['right_wrist_img'] = cv2.resize(sensor_msg.rightWristCameraRGB, size=self.resize_shape)
        obs_dict['right_gripper1_img'] = cv2.resize(sensor_msg.rightGripperCameraRGB1, size=self.resize_shape)
        obs_dict['right_gripper2_img'] = cv2.resize(sensor_msg.rightGripperCameraRGB2, size=self.resize_shape)
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
            bson_dict = BSON(bson_data).decode()
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

    def extract_msg_to_obs_dict(self, msg_path: str) -> Dict[str, np.ndarray]:
        obs_dict = dict()
        t, a, g = self.read_bson(os.path.join(msg_path, "frame_data.bson"))
        a[:, 3:] = a[:, [6,3,4,5]] # convert quat format from [x, y, z, w] to [w, x, y, z]
        t = t[:, np.newaxis]
        g = g[:, np.newaxis]
        obs_dict['timestamp'] = t

        # Add independent key-value pairs for left robot
        obs_dict['left_robot_tcp_pose'] = a
        obs_dict['left_robot_gripper_width'] = g
        
        images_array = self.load_video_frames(os.path.join(msg_path, "recording.mp4"))

        # TODO: make all sensor post-processing in parallel
        obs_dict['left_wrist_img'] = []
        for image in images_array:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            obs_dict['left_wrist_img'].append(cv2.resize(rgb_image, self.resize_shape))
            # cv2.imshow('test', rgb_image)
            # cv2.imshow('tset_resized', obs_dict['left_wrist_img'][0])
            # # Wait for a key press indefinitely or for a specified amount of time
            # cv2.waitKey(0)

            # # Close all OpenCV windows
            # cv2.destroyAllWindows()
        if self.debug:
            visualize_rgb_image(obs_dict['left_wrist_img'])

        return obs_dict
