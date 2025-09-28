import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from datetime import datetime
import os

class ImageSubscriberAndRecorder(Node):
    def __init__(self):
        super().__init__('image_subscriber_recorder')
        
        # Parameters that can be overridden
        self.declare_parameter('topic', '/left_wrist_camera/color/image_raw')
        self.declare_parameter('output_dir', '~/Videos')
        self.declare_parameter('fps', 30)
        
        # Get parameters
        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.output_dir = os.path.expanduser(
            self.get_parameter('output_dir').get_parameter_value().string_value)
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        
        # Create a subscriber for the image topic
        self.subscription = self.create_subscription(
            Image,
            topic,  # Topic name from parameter
            self.image_callback,
            10)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize video writer (will be set up when recording starts)
        self.video_writer = None
        self.is_recording = False
        self.frames = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.get_logger().info(f"Subscribing to topic: {topic}")
        self.get_logger().info(f"Video will be saved to: {self.output_dir}")
        self.get_logger().info(f"Video FPS set to: {self.fps}")
        
        self.get_logger().info('Image subscriber and recorder initialized')
        self.get_logger().info('Press "q" to stop and save the video')

    def image_callback(self, msg):
        try:
            # Handle the special case where the message claims to be rgb8 but is actually JPEG encoded
            try:
                # First try to decode as a JPEG since that's what the publisher is doing
                np_arr = np.frombuffer(msg.data, np.uint8)
                current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
                
                # Check if decoding was successful
                if current_frame is None:
                    raise ValueError("Failed to decode JPEG data")
                    
            except Exception as jpeg_error:
                self.get_logger().debug(f"JPEG decoding failed, trying standard conversion: {str(jpeg_error)}")
                # Fall back to standard conversion if JPEG decoding fails
                current_frame = self.bridge.imgmsg_to_cv2(msg)
                
                # If the encoding is rgb8 but we're expecting bgr8 for OpenCV
                # if msg.encoding == 'rgb8':
                #     current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Store frame if recording
            if self.is_recording:
                self.frames.append(current_frame)
            
            # Display the frame
            cv2.imshow('Camera Feed', current_frame)
            key = cv2.waitKey(1)
            
            # Check for 'q' press to stop recording and save video
            if key == ord('q'):
                if not self.is_recording:
                    self.is_recording = True
                    self.get_logger().info('Started recording frames')
                else:
                    self.save_video()
                    self.get_logger().info('Saved video and exiting')
                    rclpy.shutdown()
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def save_video(self):
        if not self.frames:
            self.get_logger().warn('No frames captured, video will not be saved')
            return
        
        # Get timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(self.output_dir, f'camera_recording_{timestamp}.mp4')
        
        # Get frame dimensions
        height, width = self.frames[0].shape[:2]
        
        # Create video writer with proper codec
        try:
            # Try different codecs if mp4v doesn't work
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
                video_writer = cv2.VideoWriter(video_filename, fourcc, self.fps, (width, height))
                
                if not video_writer.isOpened():
                    raise Exception("Failed to open VideoWriter with mp4v codec")
                    
            except Exception as e:
                self.get_logger().warn(f"mp4v codec failed: {str(e)}. Trying XVID codec.")
                # Try XVID codec instead
                video_filename = video_filename.replace('.mp4', '.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(video_filename, fourcc, self.fps, (width, height))
                
                if not video_writer.isOpened():
                    raise Exception("Failed to open VideoWriter with XVID codec")
            
            # Write all frames to video
            frames_written = 0
            for frame in self.frames:
                video_writer.write(frame)
                frames_written += 1
            
            # Release video writer
            video_writer.release()
            
            self.get_logger().info(f'Video saved to: {video_filename}')
            self.get_logger().info(f'Saved {frames_written} frames at {self.fps} FPS')
            
        except Exception as e:
            self.get_logger().error(f'Error saving video: {str(e)}')
            # Try to save individual frames as a fallback
            try:
                frames_dir = os.path.join(self.output_dir, f'frames_{timestamp}')
                os.makedirs(frames_dir, exist_ok=True)
                self.get_logger().info(f'Saving individual frames to {frames_dir}')
                
                for i, frame in enumerate(self.frames):
                    frame_path = os.path.join(frames_dir, f'frame_{i:04d}.jpg')
                    cv2.imwrite(frame_path, frame)
                
                self.get_logger().info(f'Saved {len(self.frames)} individual frames')
            except Exception as frame_error:
                self.get_logger().error(f'Failed to save individual frames: {str(frame_error)}')
        
        cv2.destroyAllWindows()
        self.frames = []
        self.is_recording = False

def main(args=None):
    rclpy.init(args=args)
    
    image_subscriber = ImageSubscriberAndRecorder()
    
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Save video if recording was in progress
        if image_subscriber.is_recording and image_subscriber.frames:
            image_subscriber.save_video()
        
        # Clean up
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    print("Starting ROS2 Image Subscriber and Video Recorder")
    print("This script will:")
    print("1. Connect to the /left_wrist_camera/color/image_raw topic")
    print("2. Display the incoming video feed")
    print("3. When you press 'q' once, it will START recording")
    print("4. When you press 'q' again, it will STOP recording and save the video")
    print("The video will be saved to ~/Videos with timestamp in the filename")
    print("\nHandling special case where the publisher encodes as JPEG but labels as rgb8")
    
    main()
