from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import typing

try:
    # Hydra 配置的 ListConfig
    from omegaconf import ListConfig  # type: ignore
except Exception:  # pragma: no cover - 运行时无 hydra 也不影响
    ListConfig = ()  # 退化为空元组，isinstance 将返回 False


class PickControllerFailAngle(BaseController):
    """
    夹取控制器，模拟由于末端执行器角度偏差过大导致的夹取失败。
    在接近与抓取阶段对末端姿态施加固定的角度偏移，使得夹爪姿态错误。
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
        angle_offset_deg: typing.Union[float, typing.Sequence[float]] = 30.0,
        angle_offset_range_deg: typing.Optional[typing.Union[float, typing.Sequence[float]]] = None,
        angle_offset_choices_deg: typing.Optional[typing.Union[typing.Sequence[float], "ListConfig"]] = None,
        deviation_report_threshold_deg: float = 20.0,
        about_gripper_axis: bool = True,
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

        self._angle_offset_choices_deg = self._parse_angle_choices(angle_offset_choices_deg)
        self._angle_offset_range_deg = self._parse_angle_range(angle_offset_range_deg)
        self.angle_offset_deg = self._parse_angle_offset(angle_offset_deg)
        self._angle_offset_rad = np.radians(self.angle_offset_deg)
        self._deviation_report_threshold_deg = deviation_report_threshold_deg
        self._exception_fired = False
        self._exception_reported = False
        self._about_gripper_axis = about_gripper_axis

    def _parse_angle_offset(self, angle_offset_deg):
        """返回单个角度（度），绕夹爪自身轴旋转；若为序列，取第一个分量。"""
        if isinstance(angle_offset_deg, (list, tuple, np.ndarray, ListConfig)):
            arr = np.array(angle_offset_deg, dtype=float).flatten()
            if arr.size == 0:
                raise ValueError("angle_offset_deg cannot be empty")
            return float(arr[0])
        return float(angle_offset_deg)

    def _parse_angle_range(self, angle_offset_range_deg):
        """
        返回 [lo, hi] 的标量范围（度），用于绕夹爪轴的随机转角。
        写法：
        - None: 不启用随机，使用固定 angle_offset_deg
        - 标量 s: 范围 [-s, s]
        - 长度2列表 [lo, hi]: 直接作为范围
        - 其他写法取首个元素作标量
        """
        if angle_offset_range_deg is None:
            return None
        if isinstance(angle_offset_range_deg, (list, tuple, np.ndarray, ListConfig)):
            arr = np.array(angle_offset_range_deg, dtype=float).flatten()
            if arr.size == 0:
                raise ValueError("angle_offset_range_deg cannot be empty")
            if arr.size == 1:
                s = float(arr[0])
                return np.array([-s, s], dtype=float)
            if arr.size >= 2:
                lo, hi = float(arr[0]), float(arr[1])
                return np.array([lo, hi], dtype=float)
        s = float(angle_offset_range_deg)
        return np.array([-s, s], dtype=float)

    def _parse_angle_choices(self, angle_offset_choices_deg):
        """
        支持有限集合的离散角度（度）。写法：
        - None: 不启用离散集合
        - 列表/数组/ListConfig: 至少一个值，统一转为float列表
        若提供 choices，将优先于 range 与 fixed 生效。
        """
        if angle_offset_choices_deg is None:
            return None
        if isinstance(angle_offset_choices_deg, (list, tuple, np.ndarray, ListConfig)):
            arr = np.array(angle_offset_choices_deg, dtype=float).flatten()
            if arr.size == 0:
                raise ValueError("angle_offset_choices_deg cannot be empty")
            return arr.tolist()
        # 标量也接受，转换为单元素列表
        return [float(angle_offset_choices_deg)]

    def _sample_angle_offset(self):
        """按优先级采样单轴角度偏差（度）：choices > range > fixed。"""
        if self._angle_offset_choices_deg is not None:
            return float(np.random.choice(self._angle_offset_choices_deg))
        ranges = self._angle_offset_range_deg
        if ranges is None:
            return self.angle_offset_deg
        return float(np.random.uniform(ranges[0], ranges[1]))

    def _maybe_resample_angle_offset(self):
        if self._angle_offset_range_deg is None and self._angle_offset_choices_deg is None:
            return
        self.angle_offset_deg = self._sample_angle_offset()
        self._angle_offset_rad = np.radians(self.angle_offset_deg)

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

    def _apply_angle_offset(self, base_orientation: np.ndarray) -> np.ndarray:
        """
        在目标姿态上叠加固定角度偏移，返回新的错误姿态（xyzw）。
        """
        target_rot = R.from_quat(base_orientation)
        # 绕夹爪自身轴（末端工具 z 轴）旋转 angle_offset_deg
        offset_rot = R.from_rotvec(np.array([0.0, 0.0, self._angle_offset_rad]))
        misaligned_rot = target_rot * offset_rot
        return misaligned_rot.as_quat()

    def _mark_exception(self):
        if abs(self.angle_offset_deg) >= self._deviation_report_threshold_deg:
            self._exception_fired = True

    def get_angle_deviation_info(self):
        """
        返回一次性角度偏差异常信息，用于外部标注截图。
        """
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            info = {
                "type": "angle_deviation",
                "message": f"Gripper angle deviation {self.angle_offset_deg:.1f} deg (about tool axis)",
                # 使用统一 erro 后缀，配合主循环的持久错误标记
                "suffix": "erro",
                "color": (0, 0, 255),
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

        # 始终使用带有角度偏差的错误姿态
        self._misaligned_orientation = self._apply_angle_offset(end_effector_orientation)
        self._mark_exception()

        self.pre_offset_z = pre_offset_z
        self.after_offset_z = after_offset_z
        self.pre_offset_x = pre_offset_x

        target_joint_positions = self._execute_phase(
            picking_position,
            end_effector_orientation,
            self._misaligned_orientation,
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
        base_orientation,
        misaligned_orientation,
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
                # 先用正常姿态接近，避免每一步都带误差
                target_end_effector_orientation=base_orientation,
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
                # 仍保持正常姿态
                target_end_effector_orientation=base_orientation,
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
                # 在夹取落位（即将闭合夹爪）时施加误差姿态
                target_end_effector_orientation=misaligned_orientation,
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            z_distance = abs(gripper_position[2] - picking_position[2])
            if xy_distance < 0.005 and z_distance < 0.005:
                self._event += 1
                self._t = 0
            # 在夹取那一下标记角度异常
            self._mark_exception()
            return target_joint_positions

        elif self._event == 3:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 4:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            if gripper_distances is None:
                gripper_distances = self.get_gripper_distance(object_name) / get_stage_units()
            target_joint_positions[7] = gripper_distances
            target_joint_positions[8] = gripper_distances
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.target_position = picking_position.copy()
            self.target_position[2] += self.after_offset_z / get_stage_units()
            return target_joint_positions

        elif self._event == 5:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                # 抬起阶段保持误差姿态，防止姿态抖动
                target_end_effector_orientation=misaligned_orientation,
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
        # 每个episode/重置重新采样一次角度偏差（若配置了范围）
        self._maybe_resample_angle_offset()
        # 记录本次采样值（若有）
        if self._angle_offset_range_deg is not None or self._angle_offset_choices_deg is not None:
            print(f"[fail_angle] reset sampled angle_offset_deg={self.angle_offset_deg:.2f} deg")

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


