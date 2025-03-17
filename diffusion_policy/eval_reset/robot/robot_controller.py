import requests
from requests.exceptions import RequestException
import numpy as np
import time
from PyQt6.QtCore import QObject

from utils.utility import (
    create_homogeneous_matrix,
    rotation_matrix_to_euler_angles,
    quat2eulerZYX
)

class RobotController(QObject):
    def __init__(self, robot_ip, robot_port, gui):
        super().__init__()
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.gui = gui
        self.session = requests.session()

    def send_command(self, endpoint: str, data: dict = None):
        """发送HTTP请求到机器人服务器"""
        url = f"http://{self.robot_ip}:{self.robot_port}{endpoint}"
        try:
            if 'get' in endpoint:
                response = self.session.get(url)
            else:
                if 'move' in endpoint:
                    try:
                        response = self.session.post(url, json=data, timeout=0.001)
                    except requests.exceptions.ReadTimeout:
                        return dict()
                else:
                    response = self.session.post(url, json=data)
            
            if response is not None:
                response.raise_for_status()
                return response.json()
            return dict()
        except RequestException as e:
            self.gui.log_message(f"HTTP请求错误: {str(e)}", "error")
            return dict()


    def move_robot_to_position(self, transform_file, target_positions_file, position_index):
        try:
            # 获取当前TCP位姿
            current_pose = self.send_command('/get_current_tcp/left')
            if not current_pose:
                raise Exception("无法获取当前TCP位姿")

            # 解析返回的数据
            # 前3个值是位置 (x, y, z)
            position = current_pose[:3]
            # 后4个值是四元数 (x, y, z, w)
            quaternion = current_pose[3:]
            
            # 将四元数转换为欧拉角
            euler_angles = quat2eulerZYX(quaternion, degree=True)

            # 读取变换矩阵和目标位置
            transform_matrix = np.loadtxt(transform_file)
            target_positions = np.loadtxt(target_positions_file)
            
            if position_index < 1 or position_index > len(target_positions):
                self.gui.show_message("❗", f"目标位置编号应在1到{len(target_positions)}之间")
                return

            # 计算中间位置
            homogeneous_matrix_base = create_homogeneous_matrix(position, euler_angles)
            homogeneous_matrix_world = np.linalg.inv(transform_matrix) @ homogeneous_matrix_base
            homogeneous_matrix_world[2, 3] += 0.15  # 抬高15cm
            homogeneous_matrix_base_new = transform_matrix @ homogeneous_matrix_world

            # 获取中间位置的位姿
            position_new = homogeneous_matrix_base_new[:3, 3].tolist()
            rotation_matrix_new = homogeneous_matrix_base_new[:3, :3]
            euler_angles_new = rotation_matrix_to_euler_angles(rotation_matrix_new, degrees=True).tolist()

            # 获取目标位置
            selected_position = target_positions[position_index - 1]
            x, y, z, roll, pitch, yaw = selected_position

            # 构造两段式运动命令
            primitive_cmd = (
                f"MoveL(target={x} {y} {z} {roll} {pitch} {yaw} WORLD WORLD_ORIGIN, "
                f"waypoints={position_new[0]} {position_new[1]} {position_new[2]} "
                f"{euler_angles_new[0]} {euler_angles_new[1]} {euler_angles_new[2]} WORLD WORLD_ORIGIN, "
                "maxVel=0.1)"
            )

            # 发送运动命令
            self.gui.log_message(f"执行移动命令: {primitive_cmd}", "info")
            self.send_command('/execute_primitive/left', {
                'primitive_cmd': primitive_cmd
            })
        

            # if response.get('success', False):
            #     self.gui.log_message("移动命令已发送", "info")
                
                # 等待运动完成
                # while True:
                #     status = self.send_command('/get_primitive_status/left')
                #     if status.get('reached_target', False):
                #         break
                #     time.sleep(0.1)
                
                # self.gui.log_message("移动执行完成", "info")

                # 控制夹爪
            self.gui.log_message("控制夹爪", "info")
            self.send_command('/move_gripper/left', {
                    'width': 0.1,
                    'velocity': 0.1,
                    'force_limit': 10
                })
            

            #     if gripper_response.get('success', False):
            #         self.gui.log_message("夹爪控制完成", "info")
            #         self.gui.show_message("💬", "操作完成，机器人已到达目标位置。")
            #     else:
            #         self.gui.log_message("夹爪控制失败", "error")

            # else:
            #     self.gui.log_message("移动命令执行失败", "error")

        except Exception as e:
            self.gui.log_message(f"错误: {str(e)}", "error")
            self.gui.show_message("❌", f"运行错误: {str(e)}")

    def get_robot_current_position(self, target_positions_file, position_index):
        """获取当前机器人位置"""
        try:
            # 获取当前TCP位姿
            current_pose = self.send_command('/get_current_tcp/left')
            if not current_pose:
                raise Exception("无法获取当前TCP位姿")

            # 解析返回的数据
            # 前3个值是位置 (x, y, z)
            position = current_pose[:3]
            # 后4个值是四元数 (x, y, z, w)
            quaternion = current_pose[3:]
            
            # 将四元数转换为欧拉角
            euler_angles = quat2eulerZYX(quaternion, degree=True)

            self.gui.log_message("获取到的位置和姿态信息:", "info")
            self.gui.log_message(f"位置 (x, y, z): [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]", "info")
            self.gui.log_message(f"欧拉角 (roll, pitch, yaw): [{euler_angles[0]:.4f}, {euler_angles[1]:.4f}, {euler_angles[2]:.4f}]", "info")

            # 更新目标位置文件
            try:
                target_positions = np.loadtxt(target_positions_file)
                # 如果文件为空，创建一个新的数组
                if len(target_positions.shape) == 1:
                    target_positions = target_positions.reshape(1, -1)
                
                # 创建新的位置数组
                new_position = np.concatenate([position, euler_angles])
                
                # 更新指定索引的位置
                if position_index <= len(target_positions):
                    target_positions[position_index - 1] = new_position
                else:
                    # 如果索引超出范围，添加新行
                    target_positions = np.vstack([target_positions, new_position])
                
                # 保存更新后的位置
                np.savetxt(target_positions_file, target_positions, fmt='%.6f')
                self.gui.log_message(f"已更新位置 {position_index} 的数据", "info")
                self.gui.show_message("✅", "位置数据已成功更新")
                
            except Exception as e:
                self.gui.log_message(f"更新文件时发生错误: {str(e)}", "error")

        except Exception as e:
            self.gui.log_message(f"错误: {str(e)}", "error")
            self.gui.show_message("❌", f"获取位置时发生错误: {str(e)}")