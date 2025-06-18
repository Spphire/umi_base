import rclpy
import os.path as osp
from diffusion_policy.real_world.teleoperation.data_recorder import DataRecorder
import os
import re
import threading
from diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from loguru import logger
from hydra import initialize, compose
import hydra
import time

class DataRecordManager:
    def __init__(self,
                 record_base_dir: str, 
                 record_file_dir: str, 
                 record_debug: bool = False, 
                 save_to_disk: bool = True, 
                 record_file_name: str = None, 
                 ):
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            with initialize(config_path='../config', version_base="1.3"):
                self.cfg = compose(config_name="real_world_env")
        else:
            self.cfg = compose(config_name="real_world_env")

        self.record_base_dir = record_base_dir
        self.record_file_dir = record_file_dir
        self.record_debug = record_debug
        self.save_to_disk = save_to_disk
        self.record_file_name = record_file_name

        self.transforms = RealWorldTransforms(option=self.cfg.task.transforms)
        self.record_data_flag = False
        self.record_thread = None
        self.record_stop_event = threading.Event()

    def create_record_node(self, ):
        base_dir = osp.join(self.record_base_dir, self.record_file_dir)
        if not osp.exists(base_dir):
            os.makedirs(base_dir)
        
        if self.record_file_name:
            # Use the provided file name if specified
            save_path = osp.join(base_dir, self.record_file_name)
        else:
            # Generate trial filename
            existing_trials = [
                int(re.match(r"trial(\d+)\.pkl", f).group(1))
                for f in os.listdir(base_dir)
                if re.match(r"trial(\d+)\.pkl", f)
            ]
            next_trial = max(existing_trials, default=0) + 1
            save_path = osp.join(base_dir, f"trial{next_trial}.pkl")
        
        node = DataRecorder(
            self.transforms,
            save_path=save_path,
            debug=self.record_debug,
            device_mapping_server_ip=self.cfg.task.device_mapping_server.host_ip,
            device_mapping_server_port=self.cfg.task.device_mapping_server.port
        )

        return node, save_path
    
    def record_thread_func(self):
        try:
            # Initialize ROS2 only once in the thread
            if not rclpy.ok():
                rclpy.init()
            
            node, save_path = self.create_record_node()
            logger.info(f"Recording data to: {save_path}")
            
            while not self.record_stop_event.is_set():
                try:
                    rclpy.spin_once(node, timeout_sec=0.1)
                except rclpy.executors.ExternalShutdownException:
                    break
                
        except Exception as e:
            logger.error(f"Error in recording thread: {e}")
        finally:
            try:
                if self.save_to_disk:
                    node.save()
                else:
                    logger.info("Data not saved to disk, quitting program now...")

                node.destroy_node()
                
                # Only shutdown if ROS2 is still running
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def start_recording(self):
        """Start the recording process in a separate thread"""
        if self.record_data_flag:
            logger.warning("Recording is already in progress")
            return

        try:
            self.record_stop_event.clear()
            self.record_thread = threading.Thread(target=self.record_thread_func)
            self.record_thread.start()
            self.record_data_flag = True
            logger.info("Started recording data")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.record_data_flag = False
            if self.record_thread:
                self.record_stop_event.set()
                self.record_thread.join(timeout=5)
    
    def stop_recording(self):
        """Stop the recording process"""
        if not self.record_data_flag:
            logger.warning("No recording in progress to stop")
            return
        
        try:
            self.record_stop_event.set()
            if self.record_thread:
                self.record_thread.join(timeout=3)
                if self.record_thread.is_alive():
                    logger.warning("Had to force stop recording thread")
                self.record_thread = None
            logger.info("Stopped recording data")
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
        finally:
            self.record_data_flag = False
