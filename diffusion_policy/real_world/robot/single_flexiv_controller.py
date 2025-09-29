import flexivrdk

import time
from typing import List, Optional
from loguru import logger


class DummyGripper():
    class DummyGripperStates():
        def __init__(self):
            self.width = 0.10
            self.force = 0.0
    
    def __init__(self):
        self._states = self.DummyGripperStates()

    def states(self):
        return self._states

    def Grasp(self, force_limit: float):
        self._states.force = force_limit
        logger.info(f"DummyGripper: Grasping with force limit {force_limit}")

    def Move(self, width: float, velocity: float, force_limit: float):
        self._states.width = width
        self._states.force = force_limit
        logger.info(f"DummyGripper: Moving to width {width} with velocity {velocity} and force limit {force_limit}")
    
    def Stop(self):
        self._states.force = 0.0
        logger.info(f"DummyGripper: Stopping")

class FlexivController():
    def __init__(self,
                 robot_serial_number,
                 gripper_name: Optional[str] = None
                 ) -> None:
        self.DOF=7

        try:
            self.mode = flexivrdk.Mode
            self.robot = flexivrdk.Robot(robot_serial_number)
            self.gripper = flexivrdk.Gripper(self.robot)

            self.clear_fault()
            logger.info("Enabling robot ...")
            self.robot.Enable()
            if gripper_name is not None:
                logger.info(f"Enabling gripper {gripper_name} ...")
                self.gripper.Enable(gripper_name)
                self.gripper.Init()
            else:
                logger.info("No gripper name provided, using dummy gripper ...")
                self.gripper = DummyGripper()
            seconds_waited = 0
            while not self.robot.operational():
                time.sleep(1)
                seconds_waited += 1
                if seconds_waited == 10:
                    logger.warning(
                        "Still waiting for robot to become operational, please check that the robot 1) "
                        "has no fault, 2) is in [Auto (remote)] mode")
            logger.info("Robot is now operational")
            self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        except Exception as e:
            logger.error("Error occurred while connecting to robot server: %s" % str(e))
            return None

    def clear_fault(self):
        # Fault Clearing
        # ==========================================================================================
        # Check if the robot has fault
        if self.robot.fault():
            logger.warning("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            self.robot.ClearFault()
            time.sleep(2)
            # Check again
            if self.robot.fault():
                logger.error("Fault cannot be cleared, exiting ...")
                return
            logger.info("Fault on robot server is cleared")

    def get_current_robot_states(self) -> flexivrdk.RobotStates:


        return self.robot.states()

    def get_current_gripper_states(self) -> flexivrdk.GripperStates:
        return self.gripper.states()

    def get_current_gripper_force(self) -> float:
        return self.gripper.states().force

    def get_current_gripper_width(self) -> float:
        return self.gripper.states().width

    def get_current_q(self) -> List[float]:
        # 返回flexivAPI下机械臂当前joints值
        return self.robot.states().q

    def get_current_tcp(self) -> List[float]:
        # 返回flexivAPI下机械臂当前tcp值
        return self.robot.states().tcp_pose

    def move(self, target_q):
        v = [1.5]*self.DOF #速度限制
        a = [0.8]*self.DOF #加速度限制
        self.robot.SendJointPosition(
                target_q,
                [0.0]*self.DOF,
                [0.0]*self.DOF,
                v,
                a)
            
    def tcp_move(self, target_tcp):
        self.robot.SendCartesianMotionForce(
                target_tcp, 
                [0.0]*6, 
                0.5,
                1.0)

    def execute_primitive(self, primitive_name: str, input_params: dict = {}):
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive(primitive_name, input_params)
        while not self.parse_pt_terminated_or_reachedTarget(self.robot.primitive_states()):
            time.sleep(0.001)
        self.robot.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

    @staticmethod
    def parse_pt_terminated_or_reachedTarget(pt_states):
        result = False
        if "terminated" in pt_states:
            result |= pt_states["terminated"] == 1
        if "reachedTarget" in pt_states:
            result |= pt_states["reachedTarget"] == 1
        return result
