#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import hashlib
import bson
import cv2
import numpy as np
from pprint import pprint

# ================== 配置 ==================
RECORD_DIR = r"downloads/QuestTest_records/33a6bf65-97e2-4248-aa79-ba635a199f8a"


# ================== 工具函数 ==================

def load_metadata(record_dir: str):
    meta_path = os.path.join(record_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {record_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_files(record_dir: str, files: list):
    results = []
    for file_info in files:
        filename = file_info.get("filename")
        expected_sha256 = file_info.get("sha256")
        file_path = os.path.join(record_dir, filename)

        if not os.path.exists(file_path):
            results.append((filename, "MISSING", None))
            continue

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha256.update(chunk)
        actual_sha256 = sha256.hexdigest()

        status = "OK" if expected_sha256 == actual_sha256 else "MISMATCH"
        results.append((filename, status, actual_sha256))
    return results


def load_stereo_bson(bson_path: str):
    """
    Read StereoRecordData from BSON file (dict structure)
    Returns a dict of lists, each key corresponds to one field
    """
    print(f"[INFO] Loading BSON: {bson_path}")
    with open(bson_path, "rb") as f:
        raw = f.read()
    data = bson.loads(raw)

    if not isinstance(data, dict):
        raise TypeError(f"Unexpected BSON structure: {type(data)}")

    # 使用 get 获取字段，防止缺失
    stereo_data = {
        "leftCameraAccessTimestamps": data.get("leftCameraAccessTimestamps", []),
        "leftCameraUnityTimestamps": data.get("leftCameraUnityTimestamps", []),
        "rightCameraAccessTimestamps": data.get("rightCameraAccessTimestamps", []),
        "rightCameraUnityTimestamps": data.get("rightCameraUnityTimestamps", []),
        "leftCameraPoses": data.get("leftCameraPoses", []),
        "rightCameraPoses": data.get("rightCameraPoses", []),
        "leftCameraIntrinsics": data.get("leftCameraIntrinsics", []),
        "rightCameraIntrinsics": data.get("rightCameraIntrinsics", []),
        "poseConvention": data.get("poseConvention", ""),
        "intrinsicsConvention": data.get("intrinsicsConvention", "")
    }

    # 帧数可以用 leftCameraPoses 或 rightCameraPoses 的长度判断
    num_frames = {
        "left":len(stereo_data.get("leftCameraPoses", [])),
        "right":len(stereo_data.get("rightCameraPoses", []))}
    print(f"[INFO] Loaded {num_frames} frames from BSON")
    return stereo_data, num_frames

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    cap.release()
    frames_array = np.array(frames)
    return frames_array

def analyze_video(video_path: str):
    print(f"[INFO] Analyzing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info

def get_full_data(RECORD_DIR):
    record_dir = os.path.abspath(RECORD_DIR)
    if not os.path.exists(record_dir):
        raise FileNotFoundError(f"Record directory not found: {record_dir}")

    # 1. 加载 metadata.json
    metadata = load_metadata(record_dir)
    print(f"[INFO] Loaded metadata for UUID: {metadata.get('uuid')}\n")

    # 2. 校验文件 SHA256
    files = metadata.get("files", [])
    verification_results = verify_files(record_dir, files)
    print("[INFO] File verification results:")
    for filename, status, sha256 in verification_results:
        print(f"  {filename:<20} : {status}  (sha256: {sha256})")
    print()

    # 3. 读取 BSON
    bson_file = next((f.get("filename") for f in files if f.get("filename", "").endswith(".bson")), None)
    if bson_file:
        bson_path = os.path.join(record_dir, bson_file)
        bson_data, bson_count = load_stereo_bson(bson_path)
    else:
        bson_data = {}
        bson_count = None
        print("[WARN] No BSON file found")
        return None

    # 4. 分析视频
    video_files = [f.get("filename") for f in files if f.get("filename", "").endswith(".mp4")]
    for vid_file in video_files:
        video_path = os.path.join(record_dir, vid_file)
        if os.path.exists(video_path):
            info = analyze_video(video_path)
            print(f"[INFO] Video {vid_file} info: {info}")
            if info["frame_count"] == bson_count['left' if 'left' in vid_file else 'right']:
                print("[OK] Frame count matches BSON ✔")
            else:
                print(f"[WARN] Frame count mismatch ❌ {info['frame_count']} != {bson_count['left' if 'left' in vid_file else 'right']}")
                print(info,bson_count)
        else:
            print(f"[WARN] Video file not found: {vid_file}")

    # 5. 示例帧
    for vid_file in video_files:
        if "left" in vid_file:
            left_frames = load_video_frames(os.path.join(record_dir, vid_file))
            bson_data["leftCameraFrames"] = left_frames
        if "right" in vid_file:
            right_frames = load_video_frames(os.path.join(record_dir, vid_file))
            bson_data["rightCameraFrames"] = right_frames

    return bson_data

# ================== 主流程 ==================
if __name__ == "__main__":
    get_full_data(RECORD_DIR)
