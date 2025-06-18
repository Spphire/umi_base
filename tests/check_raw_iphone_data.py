import sys
import argparse
import numpy as np
from bson import BSON
import time
import cv2

def load_bson_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            bson_data = f.read()
        bson_dict = BSON(bson_data).decode()
        return bson_dict
    except Exception as e:
        print(e)
        return None

def get_numpy_arrays(data):
    timestamps = np.array(data.get('timestamps', []))
    arkit_poses = np.array(data.get('arkitPose', []))
    gripper_widths = np.array(data.get('gripperWidth', []))
    
    return timestamps, arkit_poses, gripper_widths


def read_bson(file_path):
    data = load_bson_file(file_path)
    if data is None:
        return
    
    timestamps, arkit_poses, gripper_widths = get_numpy_arrays(data)
    return timestamps, arkit_poses, gripper_widths


def load_video_frames(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None    
    frames = []    
    while True:
        # 读取一帧
        ret, frame = cap.read()        # 如果不能读取到帧，说明到达视频末尾
        if not ret:
            break        # 将 BGR 转换为 RGB（OpenCV 默认使用 BGR 颜色空间）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # 将帧添加到帧列表中
        frames.append(rgb_frame)    # 释放视频捕获对象
    cap.release()    # 将帧列表转换为 NumPy 数组
    frames_array = np.array(frames)    
    return frames_array# 使用函数加载视频帧

if __name__ == "__main__":
    t, a, g = read_bson("./data/real_pick_and_place_iphone_test/2025-04-15_16-33-35/frame_data.bson") 
    video_path = "./data/2025-04-15_15-52-03/recording.mp4"  # 替换为你的 .mp4 文件路径
    frames_array = load_video_frames(video_path)# 输出帧数和单帧的尺寸
    if frames_array is not None:
        print("Total frames:", frames_array.shape[0])
        print("Frame size:", frames_array.shape[1:])
    print(t.shape, a.shape, g.shape)
    print(t[0], t[1])
