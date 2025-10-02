from pydantic import BaseModel, Field
import numpy as np
from enum import Enum, auto
from typing import List, final

class HandMes(BaseModel):
    q: List[float]
    pos: List[float]
    quat: List[float]  # (w, qx, qy, qz)
    thumbTip: List[float]
    indexTip: List[float]
    middleTip: List[float]
    ringTip: List[float]
    pinkyTip: List[float]
    squeeze: float
    cmd: int  # control signal (deprecated now)
    # # for left controller (B, A, joystick, trigger, side_trigger)
    # # for right controller (Y, X, joystick, trigger, side_trigger)
    # buttonState: List[float]

class interpreted_cmd(BaseModel):
    recording: bool = False
    tracking: bool = False

iphone_cmd_interpreter = {
    '10': interpreted_cmd(recording=False, tracking=False), # no recording, no tracking
    '20': interpreted_cmd(recording=True, tracking=False),  # recording, no tracking
    '12': interpreted_cmd(recording=False, tracking=True),  # no recording, tracking
    '22': interpreted_cmd(recording=True, tracking=True),   # recording, tracking
}

vr_cmd_interpreter = {
    '0': interpreted_cmd(tracking=False),  # no tracking
    '2': interpreted_cmd(tracking=True),   # tracking
}

composed_cmd_interpreter = {
    'iphone': iphone_cmd_interpreter,
    'vr': vr_cmd_interpreter,
}

class UnityMes(BaseModel):
    timestamp: float
    valid: bool
    leftHand: HandMes
    rightHand: HandMes

class Arrow(BaseModel):
    start: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    end: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

class TactileSensorMessage(BaseModel):
    device_id: str
    arrows: List[Arrow]
    scale: List[float] = Field(default_factory=lambda: [0.01, 0.005, 0.005])  # meters (sphere radius, arrow x scale, arrow y scale)

class ForceSensorMessage(BaseModel):
    device_id: str
    arrow: Arrow
    scale: List[float] = Field(default_factory=lambda: [0.01, 0.005, 0.005])  # meters (sphere radius, arrow x scale, arrow y scale)

class BimanualRobotStates(BaseModel):
    leftRobotTCP: List[float] = [0.0] * 7  # (7) (x, y, z, qw, qx, qy, qz)
    rightRobotTCP: List[float] = [0.0] * 7  # (7) (x, y, z, qw, qx, qy, qz)
    leftRobotTCPVel: List[float] = [0.0] * 6  # (6) (vx, vy, vz, wx, wy, wz)
    rightRobotTCPVel: List[float] = [0.0] * 6  # (6) (vx, vy, vz, wx, wy, wz)
    leftRobotTCPWrench: List[float] = [0.0] * 6  # (6) (fx, fy, fz, mx, my, mz)
    rightRobotTCPWrench: List[float] = [0.0] * 6  # (6) (fx, fy, fz, mx, my, mz)
    leftGripperState: List[float] = [0.0] * 2  # (2) (width, force)
    rightGripperState: List[float] = [0.0] * 2  # (2) (width, force)

class MoveGripperRequest(BaseModel):
    width: float = 0.05
    velocity: float = 10.0
    force_limit: float = 5.0

class TargetTCPRequest(BaseModel):
    target_tcp: List[float]  # (7) (x, y, z, qw, qx, qy, qz)

class ActionPrimitiveRequest(BaseModel):
    primitive_name: str
    input_params: dict = {}

class SensorMessage(BaseModel):
    # TODO: adaptable for different dimensions, considering abolishing the 2-D version
    timestamp: float
    leftRobotTCP: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (x, y, z, r, p, y)
    rightRobotTCP: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (x, y, z, r, p, y)
    leftRobotTCPVel: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (vx, vy, vz, wx, wy, wz)
    rightRobotTCPVel: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (vx, vy, vz, wx, wy, wz)
    leftRobotTCPWrench: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (fx, fy, fz, mx, my, mz)
    rightRobotTCPWrench: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (fx, fy, fz, mx, my, mz)
    leftRobotGripperState: np.ndarray = Field(default_factory=lambda: np.zeros((2, ), dtype=np.float32))  # (2) gripper (width, force)
    rightRobotGripperState: np.ndarray = Field(default_factory=lambda: np.zeros((2, ), dtype=np.float32))  # (2) gripper (width, force)
    externalCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16)) # (N, 6) (x, y, z, r, g, b)
    externalCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftWristCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16))  # (N, 6) (x, y, z, r, g, b)
    leftWristCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    rightWristCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16))  # (N, 6) (x, y, z, r, g, b)
    rightWristCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    
    vCameraImage: np.ndarray = Field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    predictedFullTCPAction: np.ndarray = Field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32))  # (N, 3), (N, 6), (N, 9) or (N, 18)
    predictedFullGipperAction: np.ndarray = Field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32))  # (N, 1), (N, 2)
    predictedPartialTCPAction: np.ndarray = Field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32))  # (N, 3), (N, 6), (N, 9) or (N, 18)
    predictedPartialGipperAction: np.ndarray = Field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32))  # (N, 1), (N, 2)

    class Config:
        arbitrary_types_allowed = True

class SensorMessageList(BaseModel):
    sensorMessages: List[SensorMessage]

class SensorMode(Enum):
    single_arm_one_realsense = auto()  # default to assume only left wrist camera used
    single_arm_two_realsense = auto()
    dual_arm_two_realsense = auto()

class TeleopMode(Enum):
    left_arm_6DOF = auto()
    right_arm_6DOF = auto()
    left_arm_3D_translation = auto()
    left_arm_3D_translation_Y_rotation = auto()
    dual_arm_3D_translation = auto()
    dual_arm_6DOF = auto()
