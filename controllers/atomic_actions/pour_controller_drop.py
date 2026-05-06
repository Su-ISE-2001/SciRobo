from omni.isaac.core.controllers import BaseController
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import get_stage_units

import numpy as np
import typing
from scipy.spatial.transform import Rotation as R

class PourControllerDrop(BaseController):
    """
    倒液控制器，模拟移动试管过程中试管掉落的情况。
    机器人在成功夹取试管后，在移动到目标位置的过程中会释放试管，导致掉落。
    
    阶段：
    - State 0: Move above the target position (在此阶段释放试管)
    - State 1: Further adjust height and position (试管已掉落，继续移动)
    - State 2: Switch joint 7 to velocity mode, start pouring (试管已掉落)
    - State 3: Hold joint 7 velocity at 0, pause pouring (试管已掉落)
    - State 4: Switch joint 7 to velocity mode, pour in reverse (试管已掉落)
    - State 5: Hold joint 7 velocity at 0, finish pouring (试管已掉落)
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1,
        position_threshold: float = 0.006
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

        self._height_range_1 = (0.3, 0.4)
        self._height_range_2 = (0.1, 0.2)
        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        
        self._drop_triggered = False  # 标记是否已经触发掉落
        self._phase0_step_count = 0  # 阶段0的步数计数
        self._drop_step_threshold = 5  # 在阶段0移动5步后触发掉落
        
        self._exception_fired = False
        self._exception_reported = False
        
        return

    def forward(
        self,
        articulation_controller: ArticulationController,
        source_size: np.ndarray,
        target_position: np.ndarray,
        current_joint_velocities: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        source_name: str = None,
        pour_speed: float = None,
        target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
    ) -> ArticulationAction:
        """
        Execute one step of the controller.
        在移动过程中释放试管。

        Args:
            articulation_controller (ArticulationController): The articulation controller for the robot.
            source_size (np.ndarray): Size of the source object being poured.
            current_joint_velocities (np.ndarray): Current joint velocities of the robot.
            gripper_control: Gripper control instance
            gripper_position (np.ndarray): Current gripper position
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
        
        # 先更新时间
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        
        if self._event == 0:
            # Move above target - 在此阶段释放试管
            target_position[2] += self._random_height_1
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=target_end_effector_orientation
            )
            
            # 在移动过程中释放试管（移动一定步数后释放）
            if not self._drop_triggered:
                # 增加阶段0的步数计数
                self._phase0_step_count += 1
                
                # 调试信息（每5步打印一次）
                if self._phase0_step_count % 5 == 0:
                    print(f"阶段0移动中，步数: {self._phase0_step_count}，累积时间: {self._t:.4f}秒，阈值: {self._drop_step_threshold}步")
                
                # 使用步数来判断（更可靠）
                if self._phase0_step_count >= self._drop_step_threshold:
                    # 释放试管
                    print(f"准备释放试管... (步数: {self._phase0_step_count}, 累积时间: {self._t:.4f}秒)")
                    gripper_control.release_object()
                    self._drop_triggered = True
                    self._exception_fired = True  # 标记异常已触发
                    print(f"试管在移动过程中掉落！(步数: {self._phase0_step_count}, 累积时间: {self._t:.4f}秒)")
            
            # 如果已经释放，打开夹爪
            if self._drop_triggered:
                # 创建包含所有关节的数组（包括夹爪关节）
                # Franka 机器人有9个关节：7个机械臂 + 2个夹爪
                num_joints = 9
                full_joint_positions = [None] * num_joints
                # 复制机械臂关节位置
                if target_joints.joint_positions is not None:
                    for i in range(min(len(target_joints.joint_positions), 7)):
                        if target_joints.joint_positions[i] is not None:
                            full_joint_positions[i] = target_joints.joint_positions[i]
                # 打开夹爪
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joints = ArticulationAction(joint_positions=full_joint_positions)
            
            self._random_height_1 = np.random.uniform(*self._height_range_1)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < 0.08:
                self._event += 1
                self._t = 0
                return target_joints
                
        elif self._event == 1:
            # Further adjust height and position - 试管已掉落，继续移动
            target_position[2] += self._random_height_2 + self.object_size[2] / 2 + self.get_pickz_offset(source_name)
            target_position[1] -= self.object_size[2] / 2 - self.get_pickz_offset(source_name)
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=target_end_effector_orientation
            )
            
            # 确保夹爪保持打开状态
            if self._drop_triggered:
                # Franka 机器人有9个关节：7个机械臂 + 2个夹爪
                num_joints = 9
                full_joint_positions = [None] * num_joints
                if target_joints.joint_positions is not None:
                    for i in range(min(len(target_joints.joint_positions), 7)):
                        if target_joints.joint_positions[i] is not None:
                            full_joint_positions[i] = target_joints.joint_positions[i]
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joints = ArticulationAction(joint_positions=full_joint_positions)
            
            self._random_height_2 = np.random.uniform(*self._height_range_2)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joints
        elif self._event == 2:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 3:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 4:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = -self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 5:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

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
        self._drop_triggered = False
        self._phase0_step_count = 0
        self._exception_fired = False
        self._exception_reported = False
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
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)
    
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

    def get_drop_info(self):
        """返回掉落异常信息用于命名/标注。"""
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            return {
                "type": "drop",
                "message": "Object dropped during movement",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
        return None

