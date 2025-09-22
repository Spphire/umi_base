import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import copy
import cv2
from loguru import logger
from diffusion_policy.common.time_utils import convert_float_to_ros_time
from ros2_numpy.point_cloud2 import array_to_pointcloud2
from diffusion_policy.common.pcd_utils import random_sample_pcd
import time

class RealsenseCameraPublisher(Node):
    """
    Realsense Camera publisher class
    """
    def __init__(
        self,
        camera_serial_number: str = '036422060422',
        camera_type: str = 'D400',
        camera_name: str = 'camera_base',
        rgb_resolution: tuple = (640, 480),
        depth_resolution: tuple = (640, 480),
        fps: int = 30,
        # (0-4) decimation_filter magnitude for point cloud
        decimate: int = 2,
        enable_pcd_publisher: bool = False,
        random_sample_point_num: int = 10000,
    ):
        node_name = f'{camera_name}_publisher'
        super().__init__(node_name)
        self.camera_serial_number = camera_serial_number
        self.camera_type = camera_type
        self.camera_name = camera_name
        self.fps = fps
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.random_sample_point_num = random_sample_point_num
        self.enable_pcd_publisher = enable_pcd_publisher

        self.color_publisher = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)
        if self.enable_pcd_publisher:
            self.depth_publisher = self.create_publisher(PointCloud2, f'/{camera_name}/depth/points', 10)
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.pipeline = None
        self.timestamp_offset = None
        self.depth_scale = None
        self.prev_time = time.time()
        self.last_print_time = time.time()  # Add a variable to keep track of the last print time
        self.frame_count = 0
        self.fps_list = []
        self.frame_intervals = []
        self.last_frame_time = None
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2 ** decimate)
        self.start()

    def start(self):
        context = rs.context()
        devices = context.query_devices()
        if len(devices) == 0:
            logger.error("No connected devices found")
            raise Exception("No connected devices found")

        config = rs.config()
        is_camera_valid = False
        for device in devices:
            serial_number = device.get_info(rs.camera_info.serial_number)
            if serial_number == self.camera_serial_number:
                is_camera_valid = True
                break

        if not is_camera_valid:
            logger.error("Camera with serial number {} not found".format(self.camera_serial_number))
            raise Exception("Camera with serial number {} not found".format(self.camera_serial_number))

        config.enable_device(self.camera_serial_number)
        self.pipeline = rs.pipeline()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        assert device_product_line == self.camera_type, f'With {self.camera_name}, Camera type does not match the camera product line.'
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # report global time
        # https://github.com/IntelRealSense/librealsense/pull/3909
        self.color_sensor = device.first_color_sensor()
        self.color_sensor.set_option(rs.option.global_time_enabled, 1)

        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        if self.enable_pcd_publisher:
            config.enable_stream(rs.stream.depth, self.depth_resolution[0], self.depth_resolution[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.rgb_resolution[0], self.rgb_resolution[1], rs.format.rgb8, 30)
        self.pipeline.start(config)
        logger.debug("Camera started!")

        # capture some frames for the camera to stabilize
        logger.debug("Capturing some frames for the camera to stabilize...")
        for _ in range(self.fps):
            self.pipeline.wait_for_frames()

        # Capture initial frames to get initial timestamps
        frames = self.pipeline.wait_for_frames()

        initial_frame = frames.get_color_frame()
        if not initial_frame:
            logger.error("Failed to get initial frame")
            raise ValueError("Failed to get initial frame")

        # convert the camera timestamp to system timestamp
        initial_camera_timestamp = convert_float_to_ros_time(initial_frame.get_timestamp() / 1000)  # convert to time class in ROS
        # we assume that the internal clock of realsense is synchronized with the system clock
        initial_system_timestamp = self.get_clock().now()

        # Calculate timestamp offset
        # TODO: measure accurate latency with QR code
        self.timestamp_offset = initial_system_timestamp - initial_system_timestamp
        logger.debug(f"Timestamp offset: {self.timestamp_offset.nanoseconds / 1e6} ms")
        logger.debug("Camera is ready! Start publishing images...")

    def stop(self):
        # Stop the camera
        self.pipeline.stop()
        logger.info("Camera stopped!")

    def convert_to_system_timestamp(self, camera_timestamp: Time) -> Time:
        """
        Convert camera timestamp to system timestamp
        """
        return camera_timestamp + self.timestamp_offset

    def publish_color_image(self, color_frame: rs.composite_frame, camera_timestamp: Time):
        """
        Publish color image
        """
        color_image = copy.deepcopy(np.asanyarray(color_frame.get_data()))
        success, encoded_image = cv2.imencode('.jpg', color_image)

        # [msg]
        msg = Image()
        msg.header.stamp = self.convert_to_system_timestamp(camera_timestamp).to_msg()
        msg.header.frame_id = "camera_color_frame"
        msg.height, msg.width, _ = color_image.shape
        msg.encoding = "rgb8"
        msg.step = msg.width * 3
        if success:
            image_bytes = encoded_image.tobytes()
            msg.data = image_bytes
        else:
            logger.warning('fail to image encoding!')
            msg.data = color_image.tobytes()

        self.color_publisher.publish(msg)


    def publish_point_cloud(self, color_frame: rs.composite_frame, depth_frame: rs.composite_frame, camera_timestamp: Time):
        """
        Publish point cloud with realsense color frame and depth frame
        """
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.asarray(color_image[:, :, ::-1], order="C")

        depth_image = np.asanyarray(depth_frame.get_data())
        resized_color_image = cv2.resize(color_image, dsize=(depth_image.shape[1], depth_image.shape[0]),
                                        interpolation=cv2.INTER_AREA)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(resized_color_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1 / self.depth_scale,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                depth_intrinsics.width, depth_intrinsics.height,
                depth_intrinsics.fx, depth_intrinsics.fy,
                depth_intrinsics.ppx, depth_intrinsics.ppy)
            )

        if self.random_sample_point_num > 0:
            points, colors = random_sample_pcd(pcd, self.random_sample_point_num, return_pcd=False)
        else:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

        r, g, b = (colors * 255).T.astype(np.uint8)

        cloud_arr = np.zeros(points.shape[0], dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32),
                                                 ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
        cloud_arr['x'], cloud_arr['y'], cloud_arr['z'] = points[:, 0], points[:, 1], points[:, 2]
        cloud_arr['r'], cloud_arr['g'], cloud_arr['b'] = r, g, b

        header = Header()
        header.stamp = self.convert_to_system_timestamp(camera_timestamp).to_msg()
        header.frame_id = "camera_depth_frame"

        # Use array_to_pointcloud2 to convert the structured array to PointCloud2 message
        pointcloud_msg = array_to_pointcloud2(cloud_arr, stamp=header.stamp, frame_id=header.frame_id)

        # Publish the point cloud
        self.depth_publisher.publish(pointcloud_msg)

    def timer_callback(self):
        """
        Publish the color and depth frames
        """

        while True:
            frames = self.pipeline.wait_for_frames()
            camera_timestamp = self.get_clock().now()
            raw_color_frame = frames.get_color_frame()

            if self.enable_pcd_publisher:
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not color_frame or not depth_frame:
                    continue
            else:
                color_frame = raw_color_frame
                depth_frame = None
                if not color_frame:
                    continue

            if self.enable_pcd_publisher:
                depth_frame = self.decimate_filter.process(depth_frame)

            self.publish_color_image(raw_color_frame, camera_timestamp)
            if self.enable_pcd_publisher:
                self.publish_point_cloud(color_frame, depth_frame, camera_timestamp)

            # [calculate fps]
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            if elapsed_time >= 1.0:
                frame_rate = self.frame_count / elapsed_time
                self.fps_list.append(frame_rate)
                self.prev_time = current_time
                self.frame_count = 0

            if self.last_frame_time is not None:
                frame_interval = (current_time - self.last_frame_time) * 1000
                self.frame_intervals.append(frame_interval)
            self.last_frame_time = current_time

            if current_time - self.last_print_time >= 5:
                logger.info(f"Publishing image from {self.camera_name} at timestamp (s): {camera_timestamp.nanoseconds / 1e9}")
                self.last_print_time = current_time
            break
