import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np
import cv2
import copy
from loguru import logger
import time
import requests
import threading
from diffusion_policy.common.time_utils import convert_float_to_ros_time
from diffusion_policy.common.precise_sleep import precise_sleep
from threading import Event, Thread
from record3d import Record3DStream

class IPhoneCameraPublisher(Node):
    def __init__(self, camera_name='external_camera', 
                 rgb_resolution=(640, 480), 
                 debug=True,
                 camera_type: str = 'iphone',
                 fps: int = 30,
                 camera_index: str = '4776'):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

        self.camera_name = camera_name
        self.rgb_resolution = rgb_resolution
        self.debug = False
        self.camera_type = camera_type
        self.fps = fps
        self.camera_index = camera_index

        # Frame rate tracking variables
        self.frame_count = 0
        self.last_log_time = time.time()
        self.last_frame_time = None

        node_name = f'{camera_name}_publisher'
        super().__init__(node_name)

        self.color_publisher_ = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)
        
        if self.debug:
            # import debugpy
            # debugpy.listen(5679)
            # print("Waiting for debugger attach.")
            # debugpy.wait_for_client()
            self.launch_debug_publisher()
        else:
            self.connect_to_device(dev_idx=0)
            self.start_processing_stream()

    def launch_debug_publisher(self, 
                               zarr_path: str = "/home/fangyuan/Documents/umi_base/.cache/cloud_pick_and_place_image_dataset/6adadfda53b00c3f/replay_buffer.zarr"):
        """
        Launch a debug publisher to publish images from a zarr file.
        """
        import zarr
        import numpy as np

        zarr_data = zarr.open(zarr_path, mode='r')
        pass

        imgs = zarr_data['data']['left_wrist_img']

        # 添加用户交互说明
        logger.info("Debug publisher started. Controls:")
        logger.info("  - Press Enter: Next image")
        logger.info("  - Type 'q' + Enter: Quit")
        logger.info("  - Type 'r' + Enter: Repeat current image")

        import keyboard

        for i in range(len(imgs)):
            img = imgs[i]
            img = np.array(img, dtype=np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Publish the RGB image
            self.debug_image = copy.deepcopy(np.array(img))

            if i == 0:
                debug_publisher_thread = Thread(target=self.debug_publisher_worker, daemon=True)
                debug_publisher_thread.start()

            logger.info(f"Publishing debug image {i + 1}/{len(imgs)}")
            # 阻塞等待用户输入
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            

        
    def debug_publisher_worker(self):
        """
        Thread to publish images for debugging purposes.
        """
        target_interval = 1.0 / 30  # Target time between frames
        next_frame_time = time.time()
        
        while True:
            current_time = time.time()
            
            if self.debug_image is not None:
                # Publish the image
                self.publish_color_image(self.debug_image, camera_timestamp=self.get_clock().now().to_msg())
                
                # Track frame rate
                self.frame_count += 1
                if self.last_frame_time is not None:
                    frame_interval = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                # Log frequency and resolution every second
                if current_time - self.last_log_time >= 1.0:
                    fps = self.frame_count / (current_time - self.last_log_time)
                    height, width = self.debug_image.shape[:2]
                    logger.info(f"[{self.camera_name}] Publishing at {fps:.2f} FPS, Resolution: {width}x{height}")
                    
                    # Reset counters
                    self.frame_count = 0
                    self.last_log_time = current_time
                
                # Calculate next frame time and sleep accordingly
                next_frame_time += target_interval
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    precise_sleep(sleep_time)
                else:
                    # If we're behind schedule, reset the timing
                    next_frame_time = time.time()
            else:
                time.sleep(0.1)

    def publish_color_image(self, color_frame, camera_timestamp):
        """
        Publish color image immediately.
        """
        current_time = time.time()
        
        # Track frame rate
        self.frame_count += 1
        if self.last_frame_time is not None:
            frame_interval = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Log frequency and resolution every second
        if current_time - self.last_log_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_log_time)
            height, width = color_frame.shape[:2]
            logger.info(f"[{self.camera_name}] Publishing at {fps:.2f} FPS, Resolution: {width}x{height}")
            
            # Reset counters
            self.frame_count = 0
            self.last_log_time = current_time
        
        color_image = copy.deepcopy(np.array(color_frame))
        success, encoded_image = cv2.imencode('.jpg', color_image)
        
        # Fill the message
        msg = Image()
        # msg.header.stamp = camera_timestamp.to_msg() # This implementation will lead to NO Obs error!
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_frame"
        msg.height, msg.width, _ = color_image.shape
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        if success:
            image_bytes = encoded_image.tobytes()
            msg.data = image_bytes
        else:
            logger.debug('fail to image encoding!')
            msg.data = color_image.tobytes()
        
        # msg = numpy_to_image(color_image, "bgr8")
        self.color_publisher_.publish(msg)

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])

    def start_processing_stream(self):
        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            # depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            # confidence = self.session.get_confidence_frame()
            # intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            camera_pose = self.session.get_camera_pose()  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

            # print(intrinsic_mat)

            # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.

            # Postprocess it
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                # depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Show the RGBD Stream
            # cv2.imshow('RGB', rgb)
            # cv2.imshow('Depth', depdebugth)

            # if confidence.shape[0] > 0 and confidence.shape[1] > 0:
            #     cv2.imshow('Confidence', confidence * 100)

            # cv2.waitKey(1)

            # Publish the RGB image
            if rgb is not None:
                self.publish_color_image(rgb, camera_timestamp=self.get_clock().now().to_msg())

            self.event.clear()

    def stop(self):
        """
        Stop the camera session.
        """
        if self.session:
            self.session.stop_stream()

def main(args=None):
    rclpy.init(args=args)
    node = IPhoneCameraPublisher(camera_name='external_camera', rgb_resolution=(640, 480), debug=False)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Shutting down node.")
    except Exception as e:
        logger.exception(e)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()