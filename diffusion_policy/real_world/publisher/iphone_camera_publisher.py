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

class IPhoneCameraPublisher(Node):
    """
    iPhone Camera publisher class optimized to publish frames immediately upon receipt.
    """
    def __init__(self,
                 camera_index: str = '4776',
                 camera_type: str = 'iphone',
                 camera_name: str = 'left_wrist_camera',
                 rgb_resolution: tuple = (640, 480),
                 fps: int = 30,
                 debug: bool = False
                 ):
        node_name = f'{camera_name}_publisher'
        super().__init__(node_name)
        self.camera_id = camera_index
        self.camera_type = camera_type
        self.camera_name = camera_name
        self.rgb_resolution = rgb_resolution
        self.debug = debug
        self.last_frame_time = time.time()

        self.color_publisher_ = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)

        self.running = True

        # Start the frame fetching thread
        self.fetch_thread = threading.Thread(target=self.fetch_frames, daemon=True)
        self.fetch_thread.start()

        logger.debug("Camera is ready! Start publishing images...")

    def fetch_frames(self):
        """
        Continuously fetch frames from the HTTP video feed in a separate thread and publish immediately.
        """
        url = 'http://localhost:1280/video-feed'  # 根据服务器地址调整
        boundary = '--frame'
        bytes_buffer = b''

    # try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            logger.error(f"无法连接到服务器，状态码：{response.status_code}")
            return

        self.last_frame_time = time.time()

        for chunk in response.iter_content(chunk_size=1024):
            if not self.running:
                break
            bytes_buffer += chunk

            while True:
                # 查找边界
                boundary_index = bytes_buffer.find(boundary.encode())
                if boundary_index == -1:
                    break
                # 查找下一个部分的边界
                next_boundary = bytes_buffer.find(boundary.encode(), boundary_index + len(boundary))
                if next_boundary == -1:
                    break
                # 提取当前部分
                part = bytes_buffer[boundary_index:next_boundary]
                bytes_buffer = bytes_buffer[next_boundary:]

                # 解析头部
                header_end = part.find(b'\r\n\r\n')
                if header_end == -1:
                    continue
                headers = part[:header_end].decode('utf-8', errors='ignore').split('\r\n')
                frame_count = None
                timestamp = None
                jpeg_data = None

                for header in headers:
                    if header.startswith('X-Frame-Count:'):
                        try:
                            frame_count = int(header.split(':')[1].strip())
                        except ValueError:
                            frame_count = None
                    elif header.startswith('X-Frame-Timestamp:'):
                        try:
                            # 假设时间戳以浮点数形式发送（例如 Unix 时间戳）
                            timestamp = float(header.split(':')[1].strip())
                        except ValueError:
                            timestamp = None

                if frame_count is not None and timestamp is not None:
                    # 提取 JPEG 数据
                    jpeg_start = header_end + 4
                    jpeg_data = part[jpeg_start:].rstrip(b'\r\n')
                    if jpeg_data:
                        # 解码 JPEG 数据为图像
                        img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            if (timestamp - self.last_frame_time) > 0.01:
                                fps = 1 / (timestamp - self.last_frame_time)
                                logger.info(f'Received frame: {frame_count}, image shape:{img.shape}, FPS: {fps:.2f}')
                            self.last_frame_time = timestamp
                            # 发布图像，传递时间戳
                            timestamp = convert_float_to_ros_time(timestamp)
                            self.publish_color_image(img, timestamp)
                            time.sleep(1./35)
        # except requests.exceptions.RequestException as e:
        #     logger.error(f"请求异常: {e}")
        # except Exception as e:
        #     logger.error(f"发生错误: {e}")

        #     print("Connection closed")
        # except requests.exceptions.RequestException as e:
        #     self.get_logger().error(f"Error fetching frames: {e}")
        # finally:
        #     self.running = False

    def publish_color_image(self, color_frame, camera_timestamp):
        """
        Publish color image immediately.
        """
        color_image = copy.deepcopy(np.array(color_frame))
        success, encoded_image = cv2.imencode('.jpg', color_image)
        
        # Fill the message
        msg = Image()
        msg.header.stamp = camera_timestamp.to_msg()
        # msg.header.stamp = self.get_clock().now().to_msg()
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

    def stop(self):
        """
        Stop the frame fetching thread.
        """
        self.running = False
        if self.fetch_thread.is_alive():
            self.fetch_thread.join(timeout=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = IPhoneCameraPublisher(camera_name='external_camera', rgb_resolution=(640, 480), debug=True)
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