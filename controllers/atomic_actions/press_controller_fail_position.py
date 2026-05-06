from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

class PressControllerFailPosition(BaseController):
    """
    按按钮控制器，模拟按按钮位置错误的情况。
    机器人在按按钮时会使用错误的位置（添加偏移），导致按按钮失败。
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        end_effector_initial_height: typing.Optional[float] = None,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_offset: float = 0.03  # 位置偏移量（米），用于模拟按按钮位置错误
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._initial_offset = initial_offset if initial_offset is not None else 0.2 / get_stage_units()
        self.position_offset = position_offset  # 位置偏移量
        
        if events_dt is None:
            self._events_dt = [0.005, 0.1, 0.01]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or NumPy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 3:
                raise Exception("events_dt length must be exactly 3")
        
        self._cspace_controller = cspace_controller
        self._start = True
        self._exception_fired = False
        self._exception_reported = False

    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        press_distance: float = 0.04
    ) -> ArticulationAction:
        """
        执行一步按按钮动作，使用错误的位置（添加偏移）。

        Args:
            target_position (np.ndarray): 目标按按钮位置。
            current_joint_positions (np.ndarray): 机器人当前关节位置。
            gripper_control: 夹爪控制器。
            end_effector_orientation (np.ndarray, optional): 末端执行器姿态。
            press_distance (float): 按压距离。

        Returns:
            ArticulationAction: 机器人控制动作。
        """
        # 确保 target_position 是有效的 numpy 数组
        if target_position is None:
            raise ValueError("target_position cannot be None")
        target_position = np.array(target_position).copy()
        
        if self._start:
            # 初始状态：打开夹爪
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions[7] = 0.04 / get_stage_units()
            target_joint_positions[8] = 0.04 / get_stage_units()
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if self.is_done():
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        # 执行当前阶段的动作
        if self._event == 0:
            # 阶段0：移动到目标物体前方（正常位置）
            target_position[0] -= self._initial_offset
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 1:
            # 阶段1：关闭夹爪
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.0015 / get_stage_units()
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 2:
            # 阶段2：向前按压到目标位置（添加位置偏移，导致按按钮位置错误）
            # 在Y方向添加偏移，模拟按按钮位置错误
            target_position[0] += press_distance / get_stage_units()
            target_position[1] += self.position_offset / get_stage_units()  # 添加Y方向偏移
            # 标记位置偏差异常
            self._exception_fired = True
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        
        return target_joint_positions

    def reset(
        self,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None
    ) -> None:
        """重置状态机到初始状态。

        Args:
            initial_offset (float, optional): 新的初始偏移距离。
            events_dt (list of float, optional): 新的阶段持续时间列表。
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if initial_offset is not None:
            self._initial_offset = initial_offset
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or NumPy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 3:
                raise Exception("events_dt length must be exactly 3")
        self._start = True
        self._exception_fired = False
        self._exception_reported = False

    def get_fail_position_info(self):
        """返回一次性按压位置偏差异常信息，用于截图/命名。"""
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            return {
                "type": "press_fail_position",
                "message": f"Press position offset {self.position_offset*1000:.1f}mm",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
        return None

    def is_done(self) -> bool:
        """检查状态机是否完成。

        Returns:
            bool: 如果达到最终阶段则返回 True，否则返回 False。
        """
        return self._event >= len(self._events_dt)

