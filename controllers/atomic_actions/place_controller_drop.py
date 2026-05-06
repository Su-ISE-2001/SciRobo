from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

class PlaceControllerDrop(BaseController):
    """
    放置控制器，模拟移动烧杯过程中烧杯滑落的情况。
    机器人在成功夹取烧杯后，在移动到目标位置的过程中会释放烧杯，导致滑落。
    
    阶段：
    - Phase 0: Move above target (在此阶段释放烧杯)
    - Phase 1: Lower to place position (烧杯已滑落，继续移动)
    - Phase 2: Wait
    - Phase 3: Open gripper (烧杯已滑落)
    - Phase 4: Lift and move away (烧杯已滑落)
    - Phase 5: Complete
    """
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        _position_threshold: float = 0.01
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0

        self._events_dt = events_dt
        if events_dt is None:
            self._events_dt = [0.005, 0.01, 0.08, 0.05, 0.01, 0.1]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt must be numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 6:
                raise Exception("events dt 6")
        self._position_threshold = _position_threshold
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._start = True
        self.target_position = None
        self._drop_triggered = False  # 标记是否已经触发掉落
        self._phase0_step_count = 0  # 阶段0的步数计数
        self._drop_step_threshold = 5  # 在阶段0移动5步后触发掉落
        self._exception_fired = False
        self._exception_reported = False

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
        
        if self._start:
            self._start = False
            self._drop_triggered = False
            self._phase0_step_count = 0
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self.is_done():
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        # 先更新时间
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0
        
        if self._event == 0:
            # Move above target - 在此阶段释放烧杯
            self.target_position = place_position.copy()
            self.target_position[2] += pre_place_z / get_stage_units()
            
            # 在移动过程中释放烧杯（移动一定步数后释放）
            if not self._drop_triggered:
                # 增加阶段0的步数计数
                self._phase0_step_count += 1
                
                # 调试信息（每5步打印一次）
                if self._phase0_step_count % 5 == 0:
                    print(f"阶段0移动中，步数: {self._phase0_step_count}，累积时间: {self._t:.4f}秒，阈值: {self._drop_step_threshold}步")
                
                # 使用步数来判断（更可靠）
                if self._phase0_step_count >= self._drop_step_threshold:
                    # 释放烧杯
                    print(f"准备释放烧杯... (步数: {self._phase0_step_count}, 累积时间: {self._t:.4f}秒)")
                    gripper_control.release_object()
                    self._drop_triggered = True
                    self._exception_fired = True
                    print(f"烧杯在移动过程中滑落！(步数: {self._phase0_step_count}, 累积时间: {self._t:.4f}秒)")
            
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # 如果已经释放，打开夹爪
            if self._drop_triggered:
                # 创建包含所有关节的数组（包括夹爪关节）
                full_joint_positions = [None] * current_joint_positions.shape[0]
                # 复制机械臂关节位置
                if target_joint_positions.joint_positions is not None:
                    for i in range(len(target_joint_positions.joint_positions)):
                        if target_joint_positions.joint_positions[i] is not None:
                            full_joint_positions[i] = target_joint_positions.joint_positions[i]
                # 打开夹爪
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joint_positions = ArticulationAction(joint_positions=full_joint_positions)
            
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
                    self._event += 1
                    self._t = 0
        elif self._event == 1:
            # Lower to place position - 烧杯已滑落，继续移动
            self.target_position = place_position.copy()
            self.target_position[2] += place_offset_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # 确保夹爪保持打开状态
            if self._drop_triggered:
                full_joint_positions = [None] * current_joint_positions.shape[0]
                if target_joint_positions.joint_positions is not None:
                    for i in range(len(target_joint_positions.joint_positions)):
                        if target_joint_positions.joint_positions[i] is not None:
                            full_joint_positions[i] = target_joint_positions.joint_positions[i]
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joint_positions = ArticulationAction(joint_positions=full_joint_positions)
            
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position - gripper_position)
                if xy_distance < 0.02:
                    self._event += 1
                    self._t = 0
        elif self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            # Open gripper - 烧杯已滑落，但继续执行打开动作
            target_joint_positions = self._gripper.forward(action="open")
            self.target_position = place_position.copy()
            self.target_position[2] += 0.15 / get_stage_units()
            self.target_position[0] -= 0.15 / get_stage_units()
            # 如果还没有释放，现在释放
            if not self._drop_triggered:
                gripper_control.release_object()
                self._drop_triggered = True
        elif self._event == 4:
            # Lift and move away - 烧杯已滑落，继续抬起
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # 确保夹爪保持打开状态
            if self._drop_triggered:
                full_joint_positions = [None] * current_joint_positions.shape[0]
                if target_joint_positions.joint_positions is not None:
                    for i in range(len(target_joint_positions.joint_positions)):
                        if target_joint_positions.joint_positions[i] is not None:
                            full_joint_positions[i] = target_joint_positions.joint_positions[i]
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joint_positions = ArticulationAction(joint_positions=full_joint_positions)
            
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
                    self._event += 1
                    self._t = 0
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        return target_joint_positions

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._drop_triggered = False
        self._phase0_step_count = 0
        self._exception_fired = False
        self._exception_reported = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities  numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 6:
                raise Exception("events dt  6")
        return

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

    def get_drop_info(self):
        """返回一次性掉落异常信息，用于截图标注/文件命名。"""
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            return {
                "type": "drop",
                # "erro" 仅作为文件名标签使用；主循环会避免把 "erro" 叠加到图像上
                "message": "erro",
                # 避免文件名出现 periodic_erro_erro：由 persist_suffix 统一追加 erro
                "suffix": "",
                "persist_suffix": "erro",
                "color": (0, 0, 255),
            }
        return None

