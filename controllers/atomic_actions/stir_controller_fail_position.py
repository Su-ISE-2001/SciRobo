from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

class StirControllerFailPosition(BaseController):
    """
    搅拌控制器，模拟搅拌棒插入烧杯位置错误的情况。
    机器人在插入搅拌棒时会使用错误的位置（添加偏移），导致插入位置错误。
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.005,
        stir_radius: float = 0.009,
        stir_speed: float = 3.0,
        position_offset: float = 0.05  # 位置偏移量（米），用于模拟插入位置错误
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0
        self.position_offset = position_offset  # 位置偏移量
        
        if events_dt is None:
            self._events_dt = [0.004, 0.004, 0.005, 0.001, 0.004]
        else:
            if not isinstance(events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            else:
                self._events_dt = events_dt
            if len(self._events_dt) != 5:
                raise Exception(f"events_dt length must be 5, got {len(self._events_dt)}")
        
        self._cspace_controller = cspace_controller
        self._position_threshold = position_threshold
        self._stir_radius = stir_radius / get_stage_units()
        self._stir_speed = stir_speed
        self._start = True
        self._current_stir_angle = 0.0

    def forward(
        self,
        center_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        执行当前阶段，使用错误的位置（添加偏移）。

        Args:
            center_position (np.ndarray): 搅拌的参考位置
            current_joint_positions (np.ndarray): 机器人当前关节位置
            gripper_position (np.ndarray): 夹爪当前位置
            end_effector_orientation (np.ndarray, optional): 末端执行器姿态

        Returns:
            ArticulationAction: 机器人要执行的关节位置
        """
        # 确保 center_position 是有效的 numpy 数组
        if center_position is None:
            raise ValueError("center_position cannot be None")
        center_position = np.array(center_position).copy()
        
        if self._start:
            self._start = False
            self._event = 0
            self._t = 0

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if self._event >= len(self._events_dt):
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        target_joint_positions = self._execute_phase(
            center_position, gripper_position, end_effector_orientation, current_joint_positions
        )

        # 基于时间的进度作为备用
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event] 
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        return target_joint_positions

    def _execute_phase(self, center_position, gripper_position, end_effector_orientation, current_joint_positions):
        """执行当前阶段并处理转换。"""
        
        if self._event == 0:
            # 阶段0：抬起搅拌棒（正常位置）
            target_position = center_position.copy()
            target_position[2] += 0.3 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            distance = np.linalg.norm(gripper_position - target_position)
            if distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 1:
            # 阶段1：移动到烧杯上方（正常位置）
            target_position = center_position.copy()
            target_position[2] += 0.3 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 2:
            # 阶段2：降低插入烧杯（添加位置偏移，导致插入位置错误）
            # 在X和Y方向添加偏移，模拟插入位置错误
            target_position = center_position.copy()
            target_position[2] += 0.12 / get_stage_units()
            target_position[0] += self.position_offset  # 添加X方向偏移
            target_position[1] += self.position_offset * 0.8  # 添加Y方向偏移（稍小一些）
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            z_distance = abs(gripper_position[2] - target_position[2])
            if z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 3:
            # 阶段3：执行搅拌动作（在错误位置进行搅拌）
            angle_increment = self._stir_speed * 0.01
            self._current_stir_angle += angle_increment
            
            x_offset = self._stir_radius * np.cos(self._current_stir_angle)
            y_offset = self._stir_radius * np.sin(self._current_stir_angle)
            target_position = center_position.copy()
            # 保持偏移，在错误位置进行搅拌
            target_position[0] += self.position_offset + x_offset
            target_position[1] += self.position_offset * 0.8 + y_offset
            target_position[2] += 0.1 / get_stage_units()
            
            return self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 4:
            # 阶段4：从烧杯中抬起（从错误位置抬起）
            target_position = center_position.copy()
            target_position[2] += 0.2 / get_stage_units()
            # 保持X和Y方向的偏移
            target_position[0] += self.position_offset
            target_position[1] += self.position_offset * 0.8
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            z_distance = abs(gripper_position[2] - target_position[2])
            if z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        else:
            return ArticulationAction(joint_positions=[None] * len(current_joint_positions))

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """重置控制器到初始状态。

        Args:
            events_dt (list of float, optional): 每个阶段的持续时间。默认为 None。

        Raises:
            Exception: 如果 'events_dt' 不是列表或 numpy 数组。
        """
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        self._current_stir_angle = 0.0
        
        if events_dt is not None:
            if not isinstance(events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            else:
                self._events_dt = events_dt
            if len(self._events_dt) != 5:
                raise Exception(f"events_dt length must be 5, got {len(self._events_dt)}")

    def is_done(self) -> bool:
        """检查搅拌序列是否完成。

        Returns:
            bool: 如果达到最后阶段则返回 True，否则返回 False。
        """
        return self._event >= len(self._events_dt)

