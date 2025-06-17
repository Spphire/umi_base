from bson import BSON
import sys
import numpy as np
import os
from pathlib import Path

def process_bson_file(bson_path):
    with open(bson_path, 'rb') as f:
        d = f.read()
        data = BSON(d).decode()
    
    timestamps = np.array(data.get('timestamps', []))
    arkit_poses = np.array(data.get('arkitPose', []))
    gripper_poses = np.array(data.get('gripperPoses', []))

    gripper_widths = np.array(data.get('gripperWidth', []))
    
    return len(timestamps), len(arkit_poses), len(gripper_widths), len(gripper_poses)

def process_directory(directory):
    # Convert to Path object for better path handling
    root_dir = Path(directory)
    
    # Walk through all subdirectories
    for subdir in root_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        # Find all .bson files in the subdirectory
        for bson_file in subdir.glob('*.bson'):
            try:
                ts_len, pose_len, grip_len, grip_pos_len = process_bson_file(bson_file)
                if ts_len != pose_len or pose_len != grip_len or grip_len != grip_pos_len:
                    print(f"{bson_file}: timestamps={ts_len}, arkit_poses={pose_len}, gripper_widths={grip_len} gripper_poses={grip_pos_len}")
            except Exception as e:
                print(f"Error processing {bson_file}: {str(e)}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_bson_length.py <directory_path>")
        sys.exit(1)
        
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a directory")
        sys.exit(1)
        
    process_directory(directory_path)