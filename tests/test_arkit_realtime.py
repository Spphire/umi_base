#!/usr/bin/env python3
"""
Real-time visualization and jump detection for hand position and pose data
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
import threading
import numpy as np
from collections import deque
from queue import Queue
import queue
import rerun as rr

# FastAPI app initialization
app = FastAPI(title="Hand Data Visualization Server")

# Define data models
class HandMes(BaseModel):
    q: Optional[List[float]] = None
    pos: List[float] = []
    quat: List[float] = []  # (w, qx, qy, qz)
    thumbTip: Optional[List[float]] = None
    indexTip: Optional[List[float]] = None
    middleTip: Optional[List[float]] = None
    ringTip: Optional[List[float]] = None
    pinkyTip: Optional[List[float]] = None
    squeeze: float = 0.0
    cmd: int = 0

class UnityMes(BaseModel):
    timestamp: float
    valid: bool
    leftHand: Optional[HandMes] = None
    rightHand: Optional[HandMes] = None

# Global variables
receive_counter = 0
lock = threading.Lock()
fps = 0

# Data buffer
data_buffer_size = 30 * 60 * 10  # 10 minutes of data
data_queue = Queue(maxsize=3000)  # Buffer for inter-thread communication
frame_counter = 0
init_timestamp = None
init_time = None

# Data storage
class DataBuffer:
    def __init__(self, max_size):
        self.pos_x_data = deque(maxlen=max_size)
        self.pos_y_data = deque(maxlen=max_size)
        self.pos_z_data = deque(maxlen=max_size)
        self.roll_data = deque(maxlen=max_size)
        self.pitch_data = deque(maxlen=max_size)
        self.yaw_data = deque(maxlen=max_size)
        self.pos_x_jumps = deque(maxlen=max_size)
        self.pos_y_jumps = deque(maxlen=max_size)
        self.pos_z_jumps = deque(maxlen=max_size)
        self.roll_jumps = deque(maxlen=max_size)
        self.pitch_jumps = deque(maxlen=max_size)
        self.yaw_jumps = deque(maxlen=max_size)
        self.timestamp = None

data_buffer = DataBuffer(data_buffer_size)

def quaternion_to_euler(w, x, y, z):
    """Convert quaternion to Euler angles (in radians)"""
    # Roll (around X-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (around Y-axis)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp) if abs(sinp) < 1 else np.copysign(np.pi / 2, sinp)

    # Yaw (around Z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def update_visualization():
    """Update rerun visualization"""
    global init_timestamp, init_time

    if data_buffer.timestamp is not None:
        latency = time.time() - init_time - (data_buffer.timestamp - init_timestamp)
        rr.log("latency", rr.TextLog(f"Latency: {latency:.2f}s"))

    # Update position plots
    rr.log("position/x", rr.Scalars(list(data_buffer.pos_x_data)[-1]))
    rr.log("position/y", rr.Scalars(list(data_buffer.pos_y_data)[-1]))
    rr.log("position/z", rr.Scalars(list(data_buffer.pos_z_data)[-1]))

    # Update rotation plots
    rr.log("rotation/roll", rr.Scalars(list(data_buffer.roll_data)[-1]))
    rr.log("rotation/pitch", rr.Scalars(list(data_buffer.pitch_data)[-1]))
    rr.log("rotation/yaw", rr.Scalars(list(data_buffer.yaw_data)[-1]))

    # Log jump points
    for name, jumps, data in [
        ("position/x/jumps", data_buffer.pos_x_jumps, data_buffer.pos_x_data),
        ("position/y/jumps", data_buffer.pos_y_jumps, data_buffer.pos_y_data),
        ("position/z/jumps", data_buffer.pos_z_jumps, data_buffer.pos_z_data),
        ("rotation/roll/jumps", data_buffer.roll_jumps, data_buffer.roll_data),
        ("rotation/pitch/jumps", data_buffer.pitch_jumps, data_buffer.pitch_data),
        ("rotation/yaw/jumps", data_buffer.yaw_jumps, data_buffer.yaw_data),
    ]:
        if jumps:
            points = np.array([[j, list(data)[j]] for j in jumps])
            rr.log(name, rr.Points2D(points))

@app.post("/unity")
async def receive_pose(data: UnityMes):
    """Receive and process Unity data"""
    global receive_counter, frame_counter, init_timestamp, init_time
    with lock:
        receive_counter += 1

    if data.leftHand and len(data.leftHand.pos) > 0:
        frame_counter += 1

        if init_timestamp is None:
            init_timestamp = data.timestamp
        if init_time is None:
            init_time = time.time()
        
        position = data.leftHand.pos
        quaternion = data.leftHand.quat
        euler = quaternion_to_euler(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        
        try:
            data_queue.put({
                'position': position,
                'euler': euler,
                'timestamp': data.timestamp,
            }, block=False)
        except Exception as e:
            print(f"Queue is full: {e}")

    return {"status": "success"}

@app.get("/")
async def root():
    """Root route"""
    return {
        "name": "Hand Data Visualization Server",
        "version": "1.0.0",
        "status": "running"
    }

def server_thread():
    """Thread function for running FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=4077, log_level='error')

def main():
    """Main function for visualization and data processing"""
    print("Starting Hand Data Visualization Server...")
    print("Listening on port: 4077")
    print("Press Ctrl+C to terminate server")
    
    # Initialize rerun
    rr.init("Hand Data Visualization", spawn=True)
    
    # Start server in a separate thread
    server_thread_instance = threading.Thread(target=server_thread, daemon=True)
    server_thread_instance.start()
    
    # Main visualization loop
    while True:
        try:
            try:
                data = data_queue.get(timeout=0.1)
                
                position = data['position']
                euler = data['euler']
                timestamp = data['timestamp']

                # print(f'timestamp:{timestamp}, current time:{time.time()}')

                data_buffer.timestamp = timestamp

                # Update position data
                for data_list, new_value in zip(
                    [data_buffer.pos_x_data, data_buffer.pos_y_data, data_buffer.pos_z_data], 
                    position
                ):
                    data_list.append(new_value)
                
                # Update pose data
                for data_list, new_value in zip(
                    [data_buffer.roll_data, data_buffer.pitch_data, data_buffer.yaw_data], 
                    euler
                ):
                    data_list.append(new_value)

                # Detect jumps
                all_data = [data_buffer.pos_x_data, data_buffer.pos_y_data, data_buffer.pos_z_data,
                           data_buffer.roll_data, data_buffer.pitch_data, data_buffer.yaw_data]
                all_jumps = [data_buffer.pos_x_jumps, data_buffer.pos_y_jumps, data_buffer.pos_z_jumps,
                            data_buffer.roll_jumps, data_buffer.pitch_jumps, data_buffer.yaw_jumps]

                threshold_xyz = 0.1
                threshold_rpy = 0.2

                for idx, (data_list, jumps) in enumerate(zip(all_data, all_jumps)):
                    if len(data_list) >= 2:
                        threshold = threshold_xyz if idx < 3 else threshold_rpy
                        if abs(data_list[-1] - data_list[-2]) > threshold:
                            jumps.append(len(data_list) - 1)

                # Update visualization
                update_visualization()

            except queue.Empty:
                pass

        except KeyboardInterrupt:
            print("\nShutting down server...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            continue

if __name__ == "__main__":
    main()