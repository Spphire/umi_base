import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from loguru import logger
import time
import cv2
import time as tm
import os

from diffusion_policy.common.time_utils import convert_float_to_ros_time


class UsbCaptureCardPublisher(Node):
    '''
    Usb Camera publisher Class
    '''

    def __init__(self,
                 camera_index: int = 0,
                 camera_type: str = 'USB',
                 fps: int = 30,
                 exposure: int = -6,
                 contrast: int = 100,
                 camera_name: str = 'left_gripper_camera_1',
                 debug=False,
                 image_folder='data/realworld',
                 recorded=False,
                 record_to_disk=False,
                 record_dir='data/recordings'
                 ):
        node_name = f'{camera_name}_publisher_{camera_index}'
        super().__init__(node_name)
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.cap = None
        self.img = None
        self.fps = fps
        self.contrast = contrast
        self.exposure = exposure
        self.width = 1920
        self.height = 1080
        self.debug = debug
        self.recorded = recorded
        
        # Recording settings
        self.record_to_disk = record_to_disk
        self.record_dir = record_dir
        if self.record_to_disk:
            # Create recording directory with timestamp
            self.timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.recording_path = os.path.join(self.record_dir, f"{self.camera_name}_{self.timestamp}")
            os.makedirs(self.recording_path, exist_ok=True)
            logger.info(f"Recording images to {self.recording_path}")
            self.frame_counter = 0

        self.color_publisher_ = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.timestamp_offset = None

        self.last_print_time = tm.time()  # Add a variable to keep track of the last print time
        self.fps_list = []
        self.frame_intervals = []
        self.last_frame_time = None

        self.prev_time = time.time()
        self.frame_count = 0

        # start the camera
        self.start()

        if self.debug:
            # Used for test on recorded videos
            self.image_folder = image_folder
            if os.path.exists(image_folder):
                self.image_files = sorted(
                    [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith('.jpg')])
            else:
                self.image_files = []
                logger.warning(f"Image folder {self.image_folder} does not exist!")
            self.total_images = len(self.image_files)
            self.current_image_index = 0

    def start(self):
        '''
        Start the usb camera
        Usb camera has no internal time,
        so we use the time we get the frame as the initial time of the topic
        '''
        if self.recorded == False:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.camera_index)
                self.set_camera_intrisics(self.cap, self.width, self.height, self.contrast, self.exposure)

                if not self.cap.isOpened():
                    self.cap.open(self.camera_index)
                    self.set_camera_intrisics(self.cap, self.width, self.height, self.contrast, self.exposure)
                    if not self.cap.isOpened():
                        logger.error("Could not open video device")
                        raise Exception("Could not open video device")

                logger.info(f"{self.camera_name} started")
            else:
                logger.warning("Camera is already running")

    def set_camera_intrisics(self, camera, width, height, contrast, exposure):
        '''
        set the resolution, contarst and resolution of the camera
        '''
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_CONTRAST, contrast)  # contrast
        camera.set(cv2.CAP_PROP_EXPOSURE, exposure)  # exposure

        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.debug(f"Requested resolution: ({width}, {height}), Actual resolution: ({actual_width}, {actual_height})")

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Camera {self.camera_index} stopped")
            
            if self.record_to_disk:
                logger.info(f"Recording stopped. Saved {self.frame_counter} frames to {self.recording_path}")
        else:
            logger.warning("Camera is not running")

    def get_rgb_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            timestamp = self.get_clock().now()
            if not ret:
                logger.error(f"Failed to capture image from camera {self.camera_index}")
                raise Exception("Failed to capture image")
            else:
                self.img = frame
            return frame, timestamp
        else:
            logger.error("Camera is not running")
            raise Exception("Camera is not running")

    '''
    get rgb_frame from recorded videos
    '''
    def get_rgb_frame_record(self):
        if self.current_image_index >= self.total_images:
            return None, None

        image_path = self.image_files[self.current_image_index]
        color_frame = cv2.imread(image_path)
        initial_time = time.time()

        self.current_image_index += 1

        return color_frame, initial_time

    def publish_color_image(self, color_image, camera_timestamp: Time):
        '''
        publish the color image
        '''
        success, encoded_image = cv2.imencode('.jpg', color_image)

        # Fill the message
        msg = Image()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f"camera_color_frame_{self.camera_index}"
        msg.height, msg.width, _ = color_image.shape
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        if success:
            image_bytes = encoded_image.tobytes()
            msg.data = image_bytes
        else:
            logger.warning('fail to image encoding!')
            msg.data = color_image.tobytes()
        self.color_publisher_.publish(msg)

    def save_frame_to_disk(self, frame, timestamp):
        """
        Save the captured frame to disk with timestamp in filename
        """
        if not self.record_to_disk:
            return
            
        try:
            # Create filename with sequence number
            filename = f"{self.camera_name}_{self.frame_counter:06d}_{timestamp.nanoseconds}.jpg"
            filepath = os.path.join(self.recording_path, filename)
            
            # Save image
            cv2.imwrite(filepath, frame)
            self.frame_counter += 1
            
            # Log every 100 frames
            if self.frame_counter % 100 == 0:
                logger.debug(f"Saved {self.frame_counter} frames to {self.recording_path}")
        except Exception as e:
            logger.error(f"Error saving frame to disk: {e}")

    def display_video_feed(self):
        '''
        Display video feed from camera or recorded files
        '''
        if self.recorded:
            while True:
                color_frame, _ = self.get_rgb_frame_record()
                if color_frame is None:
                    logger.info("All images have been played. Exiting.")
                    break

                cv2.imshow(f"Camera {self.camera_name}", color_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        else:
            while self.cap.isOpened():
                color_frame, _ = self.get_rgb_frame()
                cv2.imshow(f"Camera {self.camera_name}", color_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()

    def timer_callback(self):
        '''
        Publish the color frames
        '''
        try:
            # Get frames based on mode (recorded or live)
            if self.recorded:
                color_frame, camera_timestamp = self.get_rgb_frame_record()
                if color_frame is None:
                    logger.info("End of recorded images")
                    return
            else:
                color_frame, camera_timestamp = self.get_rgb_frame()

            # Save frame to disk if recording is enabled
            if self.record_to_disk:
                self.save_frame_to_disk(color_frame, camera_timestamp)

            # Publish the color image
            self.publish_color_image(color_frame, camera_timestamp)

            # Calculate fps
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            if elapsed_time >= 1.0:
                frame_rate = self.frame_count / elapsed_time
                self.fps_list.append(frame_rate)
                logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
                self.prev_time = current_time
                self.frame_count = 0

            # Calculate the interval between two frames
            if self.last_frame_time is not None:
                frame_interval = (current_time - self.last_frame_time) * 1000
                self.frame_intervals.append(frame_interval)
            self.last_frame_time = current_time

            # Print info every 5 seconds
            if current_time - self.last_print_time >= 5:
                logger.info(f"Publishing image from {self.camera_name} at timestamp (s): {camera_timestamp.nanoseconds / 1e9}")
                self.last_print_time = current_time

        except Exception as e:
            logger.exception(f"Error in timer callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = UsbCaptureCardPublisher(camera_index=12, camera_name='left_gripper_camera_1',
                              contrast=100,
                              debug=False,
                              recorded=False,
                              image_folder='data/tactile_video/seq01',
                              record_to_disk=True,  # Set to True to enable recording
                              record_dir='data/recordings')
    if node.debug:
        node.display_video_feed()
    else:
        try:
            rclpy.spin(node)
        except Exception as e:
            logger.exception(e)
        finally:
            node.stop()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

