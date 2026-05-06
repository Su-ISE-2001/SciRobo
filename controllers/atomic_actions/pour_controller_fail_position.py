from omni.isaac.core.controllers import BaseController
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.utils.types import ArticulationAction

import numpy as np
import typing
from scipy.spatial.transform import Rotation as R

class PourControllerFailPosition(BaseController):
    """
    倾倒控制器，模拟倾倒位置错误的情况。
    机器人在倾倒时会使用错误的位置（添加偏移），导致倾倒失败。
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1,
        position_threshold: float = 0.006,
        position_offset: float = 0.03  # 位置偏移量（米），用于模拟倾倒位置错误
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
        self.position_offset = position_offset  # 位置偏移量

        self._height_range_1 = (0.3, 0.4)
        self._height_range_2 = (0.1, 0.2)
        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        
        # 存储当前episode使用的随机方向偏移分量
        # 初始化时也生成随机方向的偏移量，确保即使 reset() 还没调用也能使用
        # 偏移量大小（欧几里得距离）固定为 position_offset
        # 但方向完全随机，确保X和Y的正负方向都能出现
        random_angle = np.random.uniform(0, 2 * np.pi)  # 随机角度（0-360度）
        self._current_offset_x = self.position_offset * np.cos(random_angle)
        self._current_offset_y = self.position_offset * np.sin(random_angle)
        
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
        执行一步控制器，使用错误的位置（添加偏移）。

        Args:
            articulation_controller (ArticulationController): 机器人关节控制器。
            source_size (np.ndarray): 源物体（被倾倒的物体）的尺寸。
            target_position (np.ndarray): 目标倾倒位置。
            current_joint_velocities (np.ndarray): 机器人当前关节速度。
            gripper_position (np.ndarray): 夹爪当前位置。
            source_name (str): 源物体名称。
            pour_speed (float, optional): 倾倒速度。默认为 None。
            target_end_effector_orientation: 末端执行器目标姿态。

        Returns:
            ArticulationAction: 要执行的关节动作。
        """
        # 确保 target_position 是有效的 numpy 数组
        if target_position is None:
            raise ValueError("target_position cannot be None")
        target_position = np.array(target_position).copy()
        
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
            # 阶段0：移动到目标位置上方（添加部分偏移，使异常更早出现）
            target_position[2] += self._random_height_1
            # 在阶段0就添加部分偏移（50%的偏移量，但保持随机方向）
            target_position[1] += self._current_offset_y * 0.5
            target_position[0] += self._current_offset_x * 0.5
            # 标记位置偏差异常
            self._exception_fired = True
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
            # 阶段1：进一步调整高度和位置（添加位置偏移，导致倾倒位置错误）
            # 使用随机方向的偏移，但偏移量大小固定
            target_position[2] += self._random_height_2 + self.object_size[2] / 2 + self.get_pickz_offset(source_name)
            target_position[1] -= self.object_size[2] / 2 - self.get_pickz_offset(source_name)
            # 添加完整大小的随机方向偏移
            target_position[1] += self._current_offset_y
            target_position[0] += self._current_offset_x
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
            # 阶段2：开始倾倒（正速度）
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 3:
            # 阶段3：暂停倾倒（速度为0）
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 4:
            # 阶段4：反向倾倒（负速度）
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = -self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 5:
            # 阶段5：完成倾倒（速度为0）
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
        重置状态机到初始阶段。

        Args:
            events_dt (list of float, optional): 每个阶段的持续时间。默认为 None。

        Raises:
            Exception: 如果 'events_dt' 不是列表或 numpy 数组。
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
        
        # 生成随机的XY方向偏移
        # 偏移量大小（欧几里得距离）固定为 position_offset
        # 但方向完全随机，确保X和Y的正负方向都能出现
        random_angle = np.random.uniform(0, 2 * np.pi)  # 随机角度（0-360度）
        self._current_offset_x = self.position_offset * np.cos(random_angle)
        self._current_offset_y = self.position_offset * np.sin(random_angle)
        
        self._exception_fired = False
        self._exception_reported = False
        return

    def is_done(self) -> bool:
        """
        检查状态机是否到达最后阶段。

        Returns:
            bool: 如果到达最后阶段则返回 True，否则返回 False。
        """
        return self._event >= len(self._events_dt)
    
    def get_pickz_offset(self, item_name):
        """计算最终抓取位置的垂直偏移。

        Args:
            item_name (str): 要抓取的物体名称。

        Returns:
            float: 垂直偏移（米）。
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

    def get_fail_position_info(self):
        """返回位置偏差异常信息用于命名/标注。"""
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            offset_magnitude = np.sqrt(self._current_offset_x**2 + self._current_offset_y**2)
            return {
                "type": "position_error",
                "message": f"Pour position error: xy offset ({self._current_offset_x*1000:.1f}, {self._current_offset_y*1000:.1f})mm, magnitude {offset_magnitude*1000:.1f}mm",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
        return None

