from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

class PlaceControllerFailPosition(BaseController):
    """
    放置控制器，模拟放置位置错误的情况。
    机器人在放置时会使用错误的位置（添加偏移），导致放置到错误的位置。
    """
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        _position_threshold: float = 0.01,
        position_offset: float = 0.08  # 位置偏移量（米），用于模拟放置位置错误
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self.position_offset = position_offset  # 位置偏移量（偏移大小）
        self._offset_x = None  # X 方向偏移（在 reset/forward 时生成，含正负）
        self._offset_y = None  # Y 方向偏移（在 reset/forward 时生成，含正负）
        self._offset_z = None  # 不使用 Z 偏移

        self._events_dt = events_dt
        if events_dt is None:
            self._events_dt = [0.005, 0.01, 0.08, 0.05, 0.01, 0.1]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt must be numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 6:
                raise Exception("events dt  6")
        self._position_threshold = _position_threshold
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._start = True
        self.target_position = None  
        self._exception_fired = False
        self._exception_reported = False
        return

    def forward(
        self,
        place_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        gripper_position: typing.Optional[np.ndarray] = None,
        pre_place_z: float = 0.2,
        place_offset_z: float = 0.05,
    ) -> ArticulationAction:
        """执行放置动作，使用错误的位置（添加偏移）。

        Args:
            place_position (np.ndarray): 目标放置位置。
            current_joint_positions (np.ndarray): 机器人当前关节位置。
            gripper_control: 夹爪控制器。
            end_effector_orientation (np.ndarray, optional): 末端执行器姿态。
            gripper_position (np.ndarray, optional): 夹爪当前位置。
            pre_place_z (float): 预放置Z轴偏移。
            place_offset_z (float): 放置Z轴偏移。

        Returns:
            ArticulationAction: 机器人控制动作。
        """
        # 确保 place_position 是有效的 numpy 数组
        if place_position is None:
            raise ValueError("place_position cannot be None")
        place_position = np.array(place_position).copy()
        
        if self._start:
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self.is_done():
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        if self._event == 0:
            # 阶段0：移动到目标位置上方（正常位置）
            self.target_position = place_position.copy()
            self.target_position[2] += pre_place_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
                    self._event += 1
                    self._t = 0
        elif self._event == 1:
            # 阶段1：降低到放置位置（XY 随机正负偏移）
            self.target_position = place_position.copy()
            self.target_position[2] += place_offset_z / get_stage_units()
            # 如未生成偏移，则采样随机方向的 XY 偏移
            if self._offset_x is None or self._offset_y is None:
                magnitude = abs(self.position_offset)
                angle = np.random.uniform(0, 2 * np.pi)
                self._offset_x = magnitude * np.cos(angle)
                self._offset_y = magnitude * np.sin(angle)
            self.target_position[0] += self._offset_x / get_stage_units()
            self.target_position[1] += self._offset_y / get_stage_units()
            # 标记位置偏差异常
            self._exception_fired = True
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                dist = np.linalg.norm(self.target_position - gripper_position)
                if dist < 0.02:
                    self._event += 1
                    self._t = 0
        elif self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            # 阶段3：打开夹爪释放物体（在错误位置释放）
            target_joint_positions = self._gripper.forward(action="open")
            self.target_position = place_position.copy()
            self.target_position[2] += 0.15 / get_stage_units()
            self.target_position[0] -= 0.15 / get_stage_units()
            # 保持 XY 方向的偏移（使用之前生成的偏移）
            if self._offset_x is not None:
                self.target_position[0] += self._offset_x / get_stage_units()
            if self._offset_y is not None:
                self.target_position[1] += self._offset_y / get_stage_units()
            gripper_control.release_object()
        elif self._event == 4:
            # 阶段4：抬起（从错误位置抬起）
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                dist = np.linalg.norm(self.target_position - gripper_position)
                if dist < self._position_threshold:
                    self._event += 1
                    self._t = 0
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        return target_joint_positions

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """重置控制器到初始状态。

        Args:
            events_dt (list of float, optional): 新的阶段持续时间。默认为 None。

        Raises:
            Exception: 如果 'events_dt' 不是列表或 numpy 数组。
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities  numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 6:
                raise Exception("events dt  6")
        self._start = True
        self.target_position = None
        # 重置时清除偏移，下次forward时会生成新的随机偏移
        self._offset_x = None
        self._offset_y = None
        self._offset_z = None
        self._exception_fired = False
        self._exception_reported = False
        return

    def get_place_position_error_info(self):
        """返回一次性放置位置偏差异常信息，用于截图/命名。"""
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            # 偏移模长基于 XY 分量
            if self._offset_x is not None or self._offset_y is not None:
                offset_norm = np.linalg.norm([self._offset_x or 0.0, self._offset_y or 0.0])
            else:
                offset_norm = 0.0
            return {
                "type": "place_position_error",
                "message": f"Place position offset XY ~{offset_norm*1000:.1f}mm",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
        return None

    def is_done(self) -> bool:
        """检查放置序列是否完成。

        Returns:
            bool: 如果达到最终阶段则返回 True，否则返回 False。
        """
        return self._event >= len(self._events_dt)

