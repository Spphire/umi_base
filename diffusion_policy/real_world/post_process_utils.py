import cv2
import numpy as np
import open3d as o3d
from loguru import logger
from typing import Dict
from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from diffusion_policy.common.data_models import SensorMessage, SensorMode
from diffusion_policy.common.visualization_utils import visualize_pcd_from_numpy, visualize_rgb_image
from diffusion_policy.common.space_utils import pose_6d_to_4x4matrix, pose_6d_to_pose_9d
from omegaconf import DictConfig

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
        obs_dict['left_wrist_img'] = self.resize_image_by_size(sensor_msg.leftWristCameraRGB, size=self.resize_shape)
        if self.debug:
            visualize_rgb_image(obs_dict['left_wrist_img'])

        obs_dict['external_img'] = self.resize_image_by_size(sensor_msg.externalCameraRGB, size=self.resize_shape)
        if self.debug:
            visualize_rgb_image(obs_dict['external_img'])
        if self.mode == SensorMode.single_arm_two_realsense or self.mode == SensorMode.single_arm_one_realsense:
            return obs_dict

        obs_dict['right_wrist_img'] = self.resize_image_by_size(sensor_msg.rightWristCameraRGB, size=self.resize_shape)
        obs_dict['right_gripper1_img'] = self.resize_image_by_size(sensor_msg.rightGripperCameraRGB1, size=self.resize_shape)
        obs_dict['right_gripper2_img'] = self.resize_image_by_size(sensor_msg.rightGripperCameraRGB2, size=self.resize_shape)
        if self.debug:
            visualize_rgb_image(obs_dict['right_wrist_img'])
            visualize_rgb_image(obs_dict['right_gripper1_img'])
            visualize_rgb_image(obs_dict['right_gripper2_img'])

            return obs_dict
        else:
            raise NotImplementedError

    @staticmethod
    def resize_image_by_size(image: np.ndarray, size: tuple) -> np.ndarray:
        return cv2.resize(image, size)
