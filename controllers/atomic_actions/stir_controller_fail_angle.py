from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import typing
from controllers.atomic_actions.stir_controller import StirController

class StirControllerFailAngle(StirController):
    """
    搅拌控制器，模拟插入烧杯时角度错误（以夹爪为轴旋转）的情况。
    机器人在插入烧杯时会绕夹爪Z轴旋转，导致角度异常。
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.005,
        stir_radius: float = 0.009,
        stir_speed: float = 3.0,
        angle_offset_deg: float = 15.0  # 角度偏移量（度），绕夹爪Z轴旋转
    ) -> None:
        super().__init__(name, cspace_controller, events_dt, position_threshold, stir_radius, stir_speed)
        self.angle_offset_deg = angle_offset_deg  # 角度偏移量
        self._angle_offset_rad = np.radians(angle_offset_deg)
        self._exception_fired = False
        self._exception_reported = False

    def _execute_phase(self, center_position, gripper_position, end_effector_orientation, current_joint_positions):
        """执行当前阶段并处理转换，在插入阶段添加角度偏移。"""
        
        if self._event == 0:
            # 阶段0：抬起玻璃棒（正常）
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
            # 阶段1：移动到烧杯上方（正常）
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
            # 阶段2：降低插入烧杯（添加角度偏移，以夹爪为轴旋转）
            target_position = center_position.copy()
            target_position[2] += 0.12 / get_stage_units()
            
            # 在插入时添加角度偏移，绕夹爪Z轴旋转（以夹爪为轴）
            base_rot = R.from_quat(end_effector_orientation)
            # 绕Z轴旋转角度偏移（以夹爪为轴）
            offset_rot = R.from_euler('z', self._angle_offset_rad, degrees=False)
            tilted_orientation = (base_rot * offset_rot).as_quat()
            
            # 标记角度异常
            self._exception_fired = True
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=tilted_orientation
            )
            
            z_distance = abs(gripper_position[2] - target_position[2])
            if z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 3:
            # 阶段3：执行搅拌动作（保持绕Z轴旋转的角度）
            angle_increment = self._stir_speed * 0.01
            self._current_stir_angle += angle_increment
            
            x_offset = self._stir_radius * np.cos(self._current_stir_angle)
            y_offset = self._stir_radius * np.sin(self._current_stir_angle)
            target_position = center_position.copy()
            target_position[0] += x_offset
            target_position[1] += y_offset
            target_position[2] += 0.1 / get_stage_units()
            
            # 保持绕Z轴旋转的角度
            base_rot = R.from_quat(end_effector_orientation)
            offset_rot = R.from_euler('z', self._angle_offset_rad, degrees=False)
            tilted_orientation = (base_rot * offset_rot).as_quat()
            
            return self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=tilted_orientation
            )

        elif self._event == 4:
            # 阶段4：从烧杯中抬起（保持绕Z轴旋转的角度）
            target_position = center_position.copy()
            target_position[2] += 0.2 / get_stage_units()
            
            # 保持绕Z轴旋转的角度
            base_rot = R.from_quat(end_effector_orientation)
            offset_rot = R.from_euler('z', self._angle_offset_rad, degrees=False)
            tilted_orientation = (base_rot * offset_rot).as_quat()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=tilted_orientation
            )
            
            z_distance = abs(gripper_position[2] - target_position[2])
            if z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        else:
            return ArticulationAction(joint_positions=[None] * len(current_joint_positions))

    def get_fail_angle_info(self):
        """
        返回一次性插入角度错误异常信息，用于外部标注截图。
        """
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            info = {
                "type": "stir_insert_angle_error",
                "message": f"Stir insert angle error: {self.angle_offset_deg:.1f} deg rotation about gripper axis",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
            return info
        return None

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """重置控制器到初始状态。"""
        super().reset()
        self._exception_fired = False
        self._exception_reported = False

