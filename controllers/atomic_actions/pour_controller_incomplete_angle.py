from omni.isaac.core.controllers import BaseController
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.utils.types import ArticulationAction

import numpy as np
import typing
from scipy.spatial.transform import Rotation as R

class PourControllerIncompleteAngle(BaseController):
    """
    倒液控制器，模拟倾倒角度不当的情况。
    机器人在倾倒时，倾倒角度不足，导致液体未能完全倒出。
    
    阶段：
    - State 0: Move above the target position (正常)
    - State 1: Further adjust height and position (正常)
    - State 2: Switch joint 7 to velocity mode, start pouring (倾倒角度不足)
    - State 3: Hold joint 7 velocity at 0, pause pouring (正常)
    - State 4: Switch joint 7 to velocity mode, pour in reverse (倾倒角度不足)
    - State 5: Hold joint 7 velocity at 0, finish pouring (正常)
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1,
        position_threshold: float = 0.006,
        angle_reduction: float = 0.5  # 倾倒角度减少的比例（默认减少50%）
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [dt / speed for dt in [0.002, 0.01, 0.009, 0.005, 0.009, 0.5]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            assert len(self._events_dt) == 6, "events dt need have length of 6 or less"
        self._cspace_controller = cspace_controller

        self._pour_default_speed = - 120.0 / 180.0 * np.pi
        self._position_threshold = position_threshold
        self._angle_reduction = angle_reduction  # 倾倒角度减少比例

        self._height_range_1 = (0.3, 0.4)
        self._height_range_2 = (0.1, 0.2)
        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        
        self._exception_fired = False
        self._exception_reported = False
        
        return

    def forward(
        self,
        articulation_controller: ArticulationController,
        source_size: np.ndarray,
        target_position: np.ndarray,
        current_joint_velocities: np.ndarray,
        gripper_position: np.ndarray,
        source_name: str = None,
        pour_speed: float = None,
        target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
    ) -> ArticulationAction:
        """
        Execute one step of the controller.
        在倾倒阶段使用减少的倾倒速度，导致倾倒角度不足。

        Args:
            articulation_controller (ArticulationController): The articulation controller for the robot.
            source_size (np.ndarray): Size of the source object being poured.
            current_joint_velocities (np.ndarray): Current joint velocities of the robot.
            gripper_position (np.ndarray): Current gripper position.
            source_name (str, optional): Name of the source object.
            pour_speed (float, optional): Speed for the pouring action. Defaults to None.
            target_end_effector_orientation: Target end effector orientation

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """
        self.object_size = source_size
        
        if pour_speed is None:
            self._pour_speed = self._pour_default_speed
        else:
            self._pour_speed = pour_speed
            
        if  self._event >= len(self._events_dt):
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            return ArticulationAction(joint_velocities=target_joint_velocities)
        
        if self._event == 0:
            target_position[2] += self._random_height_1
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=target_end_effector_orientation
            )
            self._random_height_1 = np.random.uniform(*self._height_range_1)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < 0.08:
                self._event += 1
                self._t = 0
                return target_joints
                
        elif self._event == 1:
            target_position[2] += self._random_height_2 + self.object_size[2] / 2 + self.get_pickz_offset(source_name)
            target_position[1] -= self.object_size[2] / 2 - self.get_pickz_offset(source_name)
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=target_end_effector_orientation
            )
            self._random_height_2 = np.random.uniform(*self._height_range_2)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joints
        elif self._event == 2:
            # 倾倒阶段：使用减少的倾倒速度，导致倾倒角度不足
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            reduced_pour_speed = self._pour_speed * self._angle_reduction
            print(f"倾倒阶段：正常倾倒速度={self._pour_speed:.4f}弧度/秒, 减少比例={self._angle_reduction:.2f}, 实际倾倒速度={reduced_pour_speed:.4f}弧度/秒")
            self._exception_fired = True
            target_joint_velocities[6] = reduced_pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 3:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 4:
            # 反向倾倒阶段：使用减少的倾倒速度，导致倾倒角度不足
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            reduced_pour_speed = -self._pour_speed * self._angle_reduction
            target_joint_velocities[6] = reduced_pour_speed
            self._exception_fired = True
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 5:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joints

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            events_dt (list of float, optional): Time duration for each phase. Defaults to None.

        Raises:
            Exception: If 'events_dt' is not a list or numpy array.
            Exception: If 'events_dt' length is greater than 3.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        self.object_size = None
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 3:
                raise Exception("events dt need have length of 3 or less")

        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        self._exception_fired = False
        self._exception_reported = False
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)
    
    def get_incomplete_angle_info(self):
        """返回倾倒角度不足的异常信息，用于标注/命名。"""
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            return {
                "type": "incomplete_angle",
                "message": f"Pour angle reduced by {self._angle_reduction*100:.0f}%",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
        return None
    
    def get_pickz_offset(self, item_name):
        """Calculates the vertical offset for the final grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "conical_bottle02": 0.03,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "beaker2": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
            "beaker": 0.02,
            "beaker_l": 0.02,
        }

        for key in offsets:
            if key in item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 5

