# utils.py
# dynamically get the currently running topics
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from sensor_msgs.msg import JointState
from diffusion_policy.real_world.device_mapping.device_mapping_server import DeviceToTopic
from loguru import logger

def get_topic_and_type(device_to_topic: DeviceToTopic):
    subs_name_type = []

    for camera_name, info in device_to_topic.realsense.items():
        logger.debug(f'camera info: {info}')
        subs_name_type.append((f'/{camera_name}/color/image_raw', Image))
        if info.topic_pointcloud != '':
            subs_name_type.append((f'/{camera_name}/depth/points', PointCloud2))

    for camera_name, info in device_to_topic.usb.items():
        subs_name_type.append((f'/{camera_name}/color/image_raw', Image))
        subs_name_type.append((f'/{camera_name}/marker_offset/information', PointCloud2))

    for camera_name, info in device_to_topic.capture_card.items():
        subs_name_type.append((f'/{camera_name}/color/image_raw', Image))

    subs_name_type.extend([
        ('/left_tcp_pose', PoseStamped),
        ('/right_tcp_pose', PoseStamped),
        ('/left_gripper_state', JointState),
        ('/right_gripper_state', JointState),
        ('/left_tcp_vel', TwistStamped),
        ('/right_tcp_vel', TwistStamped),
        ('/left_tcp_wrench', WrenchStamped),
        ('/right_tcp_wrench', WrenchStamped)
    ])

    return subs_name_type


