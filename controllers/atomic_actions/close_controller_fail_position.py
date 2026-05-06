from controllers.robot_controllers.grapper_manager import Gripper
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import typing
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from controllers.atomic_actions.close_controller import CloseController

class CloseControllerFailPosition(CloseController):
    """
    关门控制器，模拟推门位置错误的情况。
    机器人在推门时会使用错误的位置（添加偏移），导致推门失败。
    """
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        furniture_type: str = "drawer",
        door_width: float = 0.3,
        door_open_direction: str = None,
        position_offset: float = 0.05  # 位置偏移量（米），用于模拟推门位置错误
    ) -> None:
        super().__init__(name, cspace_controller, gripper, events_dt, furniture_type, door_width, door_open_direction)
        self.position_offset = position_offset  # 位置偏移量
        self._exception_fired = False
        self._exception_reported = False

    def _execute_door_phase(self, handle_position, end_effector_orientation, current_joint_positions, revolute_joint_position, gripper_position, angle = 50):
        """执行关门动作，在推门阶段添加位置偏移导致推门失败"""
        if self._event == 0:
            handle_position[0] -= 0.05
            handle_position[2] += 0.1
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
        elif self._event == 1:
            # 推门阶段：添加位置偏移，导致推门位置错误
            # 标记推门位置错误异常（只标记一次）
            if not self._exception_fired:
                self._exception_fired = True
            
            if self.position_rotation_interp_iter is None:
                # 在计算推门轨迹前添加位置偏移（Y方向偏移，模拟推门位置错误）
                offset_handle_position = handle_position.copy()
                offset_handle_position[1] += self.position_offset
                self.start_position = offset_handle_position.copy()
                if revolute_joint_position[1] > self.start_position[1]:
                    angle = -angle
                self.target_position = self.rotate_around_z_axis(self.start_position, revolute_joint_position, -angle)
                num_interpolation = int(600 * np.linalg.norm(self.start_position - self.target_position))
                alphas = np.linspace(start=0, stop=1, num=num_interpolation)[1:]
                position_rotation_interp_list = self.action_interpolation(
                    self.start_position, 
                    end_effector_orientation,
                    self.target_position,
                    self.rotate_quaternion_around_x(end_effector_orientation, -angle),
                    alphas,
                    joint_pos=revolute_joint_position
                )
                self.position_rotation_interp_iter = iter(position_rotation_interp_list)
            try:
                self.trans_interp, self.rotation_interp = next(self.position_rotation_interp_iter)
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=self.trans_interp,
                    target_end_effector_orientation=self.rotation_interp
                )
            except:
                self._event += 1
                self._t = 0
                target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 2:
            target_handle_position = handle_position.copy()
            target_handle_position[0] -= 0.2
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_handle_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - target_handle_position[:2])
            if xy_distance < 0.02:
                self._event += 1
                self._t = 0
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        return target_joint_positions

    def get_fail_position_info(self):
        """
        返回一次性推门位置错误异常信息，用于外部标注截图。
        """
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            info = {
                "type": "close_position_error",
                "message": f"Close position error: offset {self.position_offset*1000:.1f}mm",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
            return info
        return None

    def reset(self) -> None:
        """Reset controller state"""
        super().reset()
        self._exception_fired = False
        self._exception_reported = False

