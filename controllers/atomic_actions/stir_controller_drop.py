from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

class StirControllerDrop(BaseController):
    """
    搅拌控制器，模拟移动玻璃棒过程中玻璃棒掉落的情况。
    机器人在成功夹取玻璃棒后，在移动到目标位置的过程中会释放玻璃棒，导致掉落。
    
    阶段：
    - Phase 0: Lift the glass rod
    - Phase 1: Move above the beaker (在此阶段释放玻璃棒)
    - Phase 2: Lower into the beaker (玻璃棒已掉落，继续移动)
    - Phase 3: Perform stirring motion (玻璃棒已掉落)
    - Phase 4: Lift out of beaker (玻璃棒已掉落)
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.005,
        stir_radius: float = 0.009,
        stir_speed: float = 3.0,
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0
        
        if events_dt is None:
            self._events_dt = [0.004, 0.004, 0.005, 0.001, 0.004]  # 5 phases
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
        self._drop_triggered = False  # 标记是否已经触发掉落
        self._drop_step_threshold = 5  # 在阶段1移动5步后触发掉落（更可靠）
        self._phase1_step_count = 0  # 阶段1的步数计数

    def forward(
        self,
        center_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Execute current phase with position threshold and time-based backup.
        在移动过程中释放玻璃棒。

        Args:
            center_position (np.ndarray): Reference position for stirring
            current_joint_positions (np.ndarray): Current robot joint positions
            gripper_control: Gripper control instance
            gripper_position (np.ndarray): Current gripper position
            end_effector_orientation (np.ndarray, optional): End effector orientation

        Returns:
            ArticulationAction: Joint positions for robot execution
        """
        if self._start:
            self._start = False
            self._event = 0
            self._t = 0
            self._drop_triggered = False
            self._phase1_step_count = 0

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if self._event >= len(self._events_dt):
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        # Time-based progression as backup (先更新时间，这样 _execute_phase 可以使用更新后的时间)
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        target_joint_positions = self._execute_phase(
            center_position, gripper_position, end_effector_orientation, current_joint_positions, gripper_control
        )

        return target_joint_positions

    def _execute_phase(self, center_position, gripper_position, end_effector_orientation, current_joint_positions, gripper_control):
        """Execute current phase and handle transitions."""
        
        if self._event == 0:
            # Lift phase - 正常抬起
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
            # Move above beaker - 在此阶段释放玻璃棒
            target_position = center_position.copy()
            target_position[2] += 0.3 / get_stage_units()
            
            # 在移动过程中释放玻璃棒（移动一定步数后释放）
            if not self._drop_triggered:
                # 增加阶段1的步数计数
                self._phase1_step_count += 1
                
                # 调试信息（每5步打印一次）
                if self._phase1_step_count % 5 == 0:
                    print(f"阶段1移动中，步数: {self._phase1_step_count}，累积时间: {self._t:.4f}秒，阈值: {self._drop_step_threshold}步")
                
                # 使用步数来判断（更可靠）
                if self._phase1_step_count >= self._drop_step_threshold:
                    # 释放玻璃棒
                    print(f"准备释放玻璃棒... (步数: {self._phase1_step_count}, 累积时间: {self._t:.4f}秒)")
                    gripper_control.release_object()
                    self._drop_triggered = True
                    print(f"玻璃棒在移动过程中掉落！(步数: {self._phase1_step_count}, 累积时间: {self._t:.4f}秒)")
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # 如果已经释放，打开夹爪
            if self._drop_triggered:
                # 创建包含所有关节的数组（包括夹爪关节）
                full_joint_positions = [None] * current_joint_positions.shape[0]
                # 复制机械臂关节位置
                if target_joints.joint_positions is not None:
                    for i in range(len(target_joints.joint_positions)):
                        if target_joints.joint_positions[i] is not None:
                            full_joint_positions[i] = target_joints.joint_positions[i]
                # 打开夹爪
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joints = ArticulationAction(joint_positions=full_joint_positions)
            
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 2:
            # Lower into beaker - 玻璃棒已掉落，继续移动
            target_position = center_position.copy()
            target_position[2] += 0.12 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # 确保夹爪保持打开状态
            if self._drop_triggered and target_joints.joint_positions is not None:
                full_joint_positions = [None] * current_joint_positions.shape[0]
                for i in range(len(target_joints.joint_positions)):
                    if target_joints.joint_positions[i] is not None:
                        full_joint_positions[i] = target_joints.joint_positions[i]
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joints = ArticulationAction(joint_positions=full_joint_positions)
            
            z_distance = abs(gripper_position[2] - target_position[2])
            if z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        elif self._event == 3:
            # Stirring motion - 玻璃棒已掉落，继续搅拌动作
            angle_increment = self._stir_speed * 0.01
            self._current_stir_angle += angle_increment
            
            x_offset = self._stir_radius * np.cos(self._current_stir_angle)
            y_offset = self._stir_radius * np.sin(self._current_stir_angle)
            target_position = center_position.copy()
            target_position[0] += x_offset
            target_position[1] += y_offset
            target_position[2] += 0.1 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # 确保夹爪保持打开状态
            if self._drop_triggered and target_joints.joint_positions is not None:
                full_joint_positions = [None] * current_joint_positions.shape[0]
                for i in range(len(target_joints.joint_positions)):
                    if target_joints.joint_positions[i] is not None:
                        full_joint_positions[i] = target_joints.joint_positions[i]
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joints = ArticulationAction(joint_positions=full_joint_positions)
            
            return target_joints

        elif self._event == 4:
            # Lift out of beaker - 玻璃棒已掉落，继续抬起
            target_position = center_position.copy()
            target_position[2] += 0.2 / get_stage_units()
            
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            
            # 确保夹爪保持打开状态
            if self._drop_triggered and target_joints.joint_positions is not None:
                full_joint_positions = [None] * current_joint_positions.shape[0]
                for i in range(len(target_joints.joint_positions)):
                    if target_joints.joint_positions[i] is not None:
                        full_joint_positions[i] = target_joints.joint_positions[i]
                full_joint_positions[7] = 0.04 / get_stage_units()
                full_joint_positions[8] = 0.04 / get_stage_units()
                target_joints = ArticulationAction(joint_positions=full_joint_positions)
            
            z_distance = abs(gripper_position[2] - target_position[2])
            if z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                
            return target_joints

        else:
            return ArticulationAction(joint_positions=[None] * len(current_joint_positions))

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """Reset controller to initial state."""
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        self._current_stir_angle = 0.0
        self._drop_triggered = False
        self._phase1_step_count = 0
        
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
        """Check if stirring sequence is complete."""
        return self._event >= len(self._events_dt)

