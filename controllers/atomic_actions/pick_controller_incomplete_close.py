from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing


class PickControllerIncompleteClose(BaseController):
    """
    夹取控制器：在夹爪闭合阶段保留过大的指距，从而导致夹取失败。
    可以通过 scale 与 extra_gap 控制剩余张开量，并在需要时上报一次性异常信息。
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
        incomplete_close_scale: float = 1.5,
        extra_gap: float = 0.0,
        gap_report_threshold: float = 0.002,
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0

        if events_dt is None:
            self._events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 7:
                raise Exception(f"events_dt length must be 7, got {len(self._events_dt)}")

        self._cspace_controller = cspace_controller
        self._start = True
        self.object_size = None
        self._position_threshold = position_threshold
        self._robot_position = None

        self._incomplete_close_scale = max(float(incomplete_close_scale), 0.0)
        self._extra_gap = max(float(extra_gap), 0.0)
        self._gap_report_threshold = max(float(gap_report_threshold), 0.0)
        self._exception_fired = False
        self._exception_reported = False
        self._planned_gripper_distance = None
        self._failed_gripper_distance = None

    def set_robot_position(self, position: np.ndarray):
        self._robot_position = position

    def _calculate_approach_direction(self, picking_position: np.ndarray) -> np.ndarray:
        if self._robot_position is None:
            return np.array([-1, 0, 0])

        relative_pos = picking_position - self._robot_position
        horizontal_vec = relative_pos.copy()
        horizontal_vec[2] = 0

        if np.linalg.norm(horizontal_vec) > 0:
            horizontal_vec = -horizontal_vec / np.linalg.norm(horizontal_vec)
        else:
            horizontal_vec = np.array([-1, 0, 0])
        return horizontal_vec

    def _mark_exception(self):
        if (
            self._planned_gripper_distance is not None
            and self._failed_gripper_distance is not None
            and (self._failed_gripper_distance - self._planned_gripper_distance) >= self._gap_report_threshold
        ):
            self._exception_fired = True

    def get_incomplete_close_info(self):
        """
        返回一次性异常信息，用于截图或日志标记。
        """
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            info = {
                "type": "incomplete_close",
                "message": (
                    f"Gripper left gap {self._failed_gripper_distance - self._planned_gripper_distance:.3f} m "
                    f"(scale={self._incomplete_close_scale:.2f}, extra_gap={self._extra_gap:.3f} m)"
                ),
                "suffix": "incomplete_close",
                "color": (255, 0, 0),
            }
            return info
        return None

    def forward(
        self,
        picking_position: np.ndarray,
        current_joint_positions: np.ndarray,
        object_name: str,
        object_size: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        pre_offset_z: float = 0.12,
        after_offset_z: float = 0.15,
        pre_offset_x: float = 0.1,
        gripper_distances: float = None,
    ) -> ArticulationAction:
        if picking_position is None:
            raise ValueError("picking_position cannot be None")
        picking_position = np.array(picking_position)

        if object_size is None:
            raise ValueError("object_size cannot be None")
        self.object_size = np.array(object_size)

        if self._start:
            return self._handle_start_state(current_joint_positions)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self.pre_offset_z = pre_offset_z
        self.after_offset_z = after_offset_z
        self.pre_offset_x = pre_offset_x

        target_joint_positions = self._execute_phase(
            picking_position,
            end_effector_orientation,
            current_joint_positions,
            object_name,
            gripper_control,
            gripper_position,
            gripper_distances,
        )

        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        return target_joint_positions

    def _handle_start_state(self, current_joint_positions):
        self._start = False
        target_joint_positions = [None] * current_joint_positions.shape[0]
        target_joint_positions[7] = 0.04 / get_stage_units()
        target_joint_positions[8] = 0.04 / get_stage_units()
        return ArticulationAction(joint_positions=target_joint_positions)

    def _execute_phase(
        self,
        picking_position,
        end_effector_orientation,
        current_joint_positions,
        object_name,
        gripper_control,
        gripper_position,
        gripper_distances,
    ):
        approach_dir = self._calculate_approach_direction(picking_position)

        if self._event == 0:
            picking_position = picking_position + approach_dir * (self.pre_offset_x / get_stage_units())
            picking_position[2] += self.object_size[2] + self.pre_offset_z
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 1:
            picking_position = picking_position + approach_dir * (0.1 / get_stage_units())
            picking_position[2] += self.get_pickprez_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 2:
            picking_position[2] += self.get_pickz_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            z_distance = abs(gripper_position[2] - picking_position[2])
            if xy_distance < 0.005 and z_distance < 0.005:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 3:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 4:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            if gripper_distances is None:
                gripper_distances = self.get_gripper_distance(object_name) / get_stage_units()
            self._planned_gripper_distance = gripper_distances

            failed_distance = (
                gripper_distances * self._incomplete_close_scale
                + self._extra_gap / get_stage_units()
            )
            # 钳制到“最大张开量”，保证可稳定模拟“夹爪没关/不闭合”
            max_open_distance = 0.04 / get_stage_units()
            failed_distance = min(float(failed_distance), float(max_open_distance))
            self._failed_gripper_distance = failed_distance
            target_joint_positions[7] = failed_distance
            target_joint_positions[8] = failed_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.target_position = picking_position.copy()
            self.target_position[2] += self.after_offset_z / get_stage_units()
            self._mark_exception()
            return target_joint_positions

        elif self._event == 5:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation,
            )

            xy_distance = np.linalg.norm(gripper_position[:2] - self.target_position[:2])
            z_distance = abs(gripper_position[2] - self.target_position[2])
            if xy_distance < self._position_threshold and z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions
        else:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0

        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 7:
                raise Exception(f"events_dt length must be 7, got {len(self._events_dt)}")

        self._start = True
        self.object_size = None
        self._robot_position = None
        self._exception_fired = False
        self._exception_reported = False
        self._planned_gripper_distance = None
        self._failed_gripper_distance = None

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

    def get_gripper_distance(self, item_name):
        gripper_distances = {
            "rod": 0.003,
            "tube": 0.01,
            "beaker": 0.022,
            "beaker2": 0.028,
            "beaker_l": 0.03,
            "beaker_04": 0.025,
            "beaker_05": 0.025,
            "beaker_03": 0.025,
            "Erlenmeyer flask": 0.018,
            "Petri dish": 0.005,
            "pipette": 0.008,
            "microscope slide": 0.002,
            "conical_bottle01": 0.01,
            "conical_bottle02": 0.023,
            "conical_bottle03": 0.03,
            "conical_bottle04": 0.03,
            "graduated_cylinder_01": 0.005,
            "graduated_cylinder_02": 0.018,
            "graduated_cylinder_03": 0.024,
            "graduated_cylinder_04": 0.030,
            "Tampa_100mL_Lid_MAT_0": 0.014,
            "Vidro_100mL_Glass_MAT_0": 0.016,
            "BalaoVolumetrico_100mL": 0.002,
            "xform": 0.003,
            "titrationflasks": 0.020,
            "titrationflasks_01": 0.003,
        }

        for key in gripper_distances:
            if key == item_name.lower():
                return gripper_distances[key]

        return 0.01

    def get_pickz_offset(self, item_name):
        offsets = {
            "conical_bottle02": 0.04,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "beaker": 0.0,
            "beaker_04": 0.0,
            "beaker_05": 0.0,
            "beaker_03": 0.0,
            "beaker2": 0.0,
            "beaker_2": 0.0,
            "beaker_l": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
            "glass_rod": 0.02,
            "xform": 0.03,
            "titrationflasks": 0.04,
            "titrationflasks_01": 0.02,
        }

        for key in offsets:
            if key == item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 5

    def get_pickprez_offset(self, item_name):
        offsets = {
            "volume_flask": 0,
            "beaker2": 0.05,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "graduated_cylinder_01": 0.05,
            "graduated_cylinder_02": 0.03,
            "graduated_cylinder_03": 0.03,
            "graduated_cylinder_04": 0.03,
            "xform": 0.05,
            "titrationflasks": 0.06,
            "titrationflasks_01": 0.04,
        }

        for key in offsets:
            if key == item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 3

