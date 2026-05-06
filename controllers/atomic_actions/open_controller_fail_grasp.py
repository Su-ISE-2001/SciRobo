from controllers.robot_controllers.grapper_manager import Gripper
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import typing
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

class OpenControllerFailGrasp(BaseController):
    """
    Open controller that simulates failure to grasp the door handle.
    The robot will approach the handle but miss it by a small offset,
    preventing successful grasping.
    """
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        events_dt: typing.Optional[typing.List[float]] = None,
        furniture_type: str = "drawer",
        door_width: float = 0.3,
        door_open_direction: str = "counterclockwise",
        grasp_offset: float = 0.03  # Offset to miss the handle (in meters)
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self.furniture_type = furniture_type
        self.door_width = door_width
        self.door_open_direction = door_open_direction
        self.position_rotation_interp_iter = None
        # Allow symmetric offset (positive or negative)
        self.grasp_offset = grasp_offset  # magnitude
        self._current_grasp_offset = self._sample_offset(grasp_offset)
        self._exception_fired = False
        self._exception_reported = False
        
        if events_dt is None:
            self._events_dt = [0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 8:
                raise Exception(f"events_dt length must be 8, got {len(self._events_dt)}")

        self._position_threshold = 0.01 / get_stage_units()
        
    def forward(
        self,
        handle_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        revolute_joint_position: np.ndarray = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        angle: float = 50.0,
        close_gripper_distance: float = 0.023
    ) -> ArticulationAction:
        """
        Execute one step of opening control with failed grasp.

        Args:
            handle_position (np.ndarray): Handle position
            current_joint_positions (np.ndarray): Current robot joint positions
            revolute_joint_position (np.ndarray): Revolute joint position
            gripper_position (np.ndarray): Gripper position
            end_effector_orientation (np.ndarray, optional): End effector orientation (quaternion).
                Defaults to None, which uses [0, np.pi, 0].

        Returns:
            ArticulationAction: Control action
        """

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat([0, 110, 0], degrees=True, extrinsic=False)
            
        self._t += self._events_dt[self._event]
        
        target_joint_positions = self._execute_phase(
            handle_position, 
            end_effector_orientation, 
            current_joint_positions,
            gripper_position,
            revolute_joint_position,
            angle,
            close_gripper_distance
        )
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions
    
    def _execute_phase(self, handle_position, end_effector_orientation, current_joint_positions, gripper_position, revolute_joint_position = None, angle = 50, close_gripper_distance = 0.023):
        """Execute current phase of grasping action"""
        if self.furniture_type == "drawer":
            return self._execute_drawer_phase(handle_position, end_effector_orientation, current_joint_positions, gripper_position)
        else:
            return self._execute_door_phase(handle_position, end_effector_orientation, current_joint_positions, revolute_joint_position, gripper_position, angle, close_gripper_distance)
    
    def _execute_drawer_phase(self, handle_position, end_effector_orientation, current_joint_positions, gripper_position):
        """Execute drawer opening action with failed grasp"""
        if self._event == 0:
            handle_position[0] -= 0.08 / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=handle_position, target_end_effector_orientation=end_effector_orientation
                )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joint_positions
        elif self._event == 1:
            # Approach handle but with offset to miss it
            handle_position[0] -= 0.015
            handle_position[1] += self._current_grasp_offset  # Add offset to miss the handle
            target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=handle_position, target_end_effector_orientation=end_effector_orientation
                )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold / 3:
                self._event += 1
                self._t = 0
                return target_joint_positions
        elif self._event == 2:
            # Try to close gripper but miss the handle
            handle_position[0] -= 0.015
            handle_position[1] += self._current_grasp_offset  # Maintain offset
            # 标记抓取失败异常
            self._exception_fired = True
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.01 / get_stage_units()
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.target_position = handle_position
            self.target_position[0] -= 0.1 / get_stage_units()
        elif self._event == 3:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 4:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 5:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.04 / get_stage_units()
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 6:
            # Move away since grasp failed
            handle_position[0] -= 0.12 / get_stage_units()
            handle_position[2] += 0.06
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            )
        else:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        return target_joint_positions

    def _execute_door_phase(self, handle_position, end_effector_orientation, current_joint_positions, revolute_joint_position, gripper_position, angle = 50, close_gripper_distance = 0.023):
        """Execute door opening action with failed grasp"""
        if self._event == 0:
            handle_position[0] -= 0.08
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joint_positions
        elif self._event == 1:
            # Approach handle but with offset to miss it
            handle_position[0] -= 0.015
            handle_position[1] += self._current_grasp_offset  # Add offset to miss the handle
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold / 3:
                self._event += 1
                self._t = 0
                return target_joint_positions
        elif self._event == 2:
            # Try to close gripper but miss the handle due to offset
            handle_position[0] -= 0.015
            handle_position[1] += self._current_grasp_offset  # Maintain offset
            # 标记抓取失败异常
            self._exception_fired = True
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions[7] = close_gripper_distance
            target_joint_positions[8] = close_gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.start_position = handle_position.copy()
        elif self._event == 3:
            # Since grasp failed, just wait and then move away
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 4:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 5:
            # Open gripper since we didn't grasp anything
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.04
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 6:
            # Move away from handle since grasp failed
            handle_position = self.start_position.copy() if hasattr(self, 'start_position') else handle_position.copy()
            handle_position[0] -= 0.08  # Move back
            handle_position[1] += 0.05  # Move away from handle
            handle_position[2] += 0.05  # Move up slightly
            
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joint_positions
        else:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        return target_joint_positions
    
    def get_fail_grasp_info(self):
        """
        返回一次性抓取失败异常信息，用于外部标注截图。
        """
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            info = {
                "type": "grasp_failure",
                "message": f"Grasp failed: offset {self._current_grasp_offset*1000:.1f}mm",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
            return info
        return None
    
    def reset(self) -> None:
        """Reset controller state"""
        BaseController.reset(self)
        self._event = 0
        self._t = 0
        self.position_rotation_interp_iter = None
        self._exception_fired = False
        self._exception_reported = False
        self._current_grasp_offset = self._sample_offset(self.grasp_offset)

    def is_done(self) -> bool:
        """Check if controller has completed all states"""
        return self._event >= len(self._events_dt)

    def _sample_offset(self, magnitude: float) -> float:
        """Sample a symmetric offset (+/- magnitude)."""
        mag = abs(float(magnitude))
        sign = np.random.choice([-1.0, 1.0])
        return sign * mag

