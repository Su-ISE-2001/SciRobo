from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

# 支持 OmegaConf 类型
try:
    from omegaconf import ListConfig, DictConfig
    _HAS_OMEGACONF = True
except ImportError:
    _HAS_OMEGACONF = False
    ListConfig = list
    DictConfig = dict

class PlaceControllerWrongAngle(BaseController):
    """
    放置控制器，模拟放置角度异常的情况。
    机器人在放置物体时，末端执行器旋转了一定角度，导致物体以错误的角度被放置。
    
    阶段：
    - State 0: Move above the target position (使用错误的角度)
    - State 1: Lower to place position (使用错误的角度)
    - State 2: Wait
    - State 3: Open gripper and release (使用错误的角度)
    - State 4: Move away (使用错误的角度)
    - State 5: Finish
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        _position_threshold: float = 0.01,
        rotation_angle: typing.Union[float, typing.List[float], typing.Dict[str, float], None] = None,
        random_rotation: bool = True,  # 是否使用随机旋转角度
        rotation_range: typing.Union[float, typing.List[float], typing.Dict[str, typing.List[float]], None] = None,
        min_abs_rotation_deg: float = 0.0,
        abs_rotation_range_deg: typing.Union[
            typing.List[float],
            typing.List[typing.List[float]],
            typing.Dict[str, typing.List[float]],
            None,
        ] = None,
        # 旋转角度，可以是：
        # - 单个浮点数：绕Z轴旋转（度），默认30度
        # - 列表 [x, y, z]：绕X、Y、Z轴旋转的角度（度）
        # - 字典 {"x": 30, "y": 20, "z": 10}：分别指定各轴旋转角度（度）
        # rotation_range: 随机旋转范围，格式与 rotation_angle 相同，但每个值表示 [min, max] 范围
        # - 单个数字：Z轴范围，例如 30.0 表示 [-30, 30]
        # - 列表 [[x_min, x_max], [y_min, y_max], [z_min, z_max]] 或 [z_min, z_max]
        # - 字典 {"x": [min, max], "y": [min, max], "z": [min, max]}
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
        self._random_rotation = random_rotation
        self._min_abs_rotation_deg = float(min_abs_rotation_deg)
        self._exception_fired = False
        self._exception_reported = False
        self._abs_rotation_ranges = None
        
        # 解析旋转角度参数（用于固定角度模式）
        self._fixed_rotation_angles = None
        if not random_rotation:
            if rotation_angle is None:
                # 默认：只绕Z轴旋转30度
                self._fixed_rotation_angles = {"x": 0.0, "y": 0.0, "z": 30.0}
            elif isinstance(rotation_angle, (int, float)):
                # 单个数字：只绕Z轴旋转
                self._fixed_rotation_angles = {"x": 0.0, "y": 0.0, "z": float(rotation_angle)}
            elif isinstance(rotation_angle, (list, tuple, np.ndarray, ListConfig)):
                # 列表：[x, y, z] 或 [z]（向后兼容）
                rotation_angle = list(rotation_angle)  # 转换为普通列表
                if len(rotation_angle) == 1:
                    self._fixed_rotation_angles = {"x": 0.0, "y": 0.0, "z": float(rotation_angle[0])}
                elif len(rotation_angle) == 3:
                    self._fixed_rotation_angles = {"x": float(rotation_angle[0]), "y": float(rotation_angle[1]), "z": float(rotation_angle[2])}
                else:
                    raise ValueError("rotation_angle list must have 1 or 3 elements")
            elif isinstance(rotation_angle, (dict, DictConfig)):
                # 字典：{"x": 30, "y": 20, "z": 10}
                rotation_angle = dict(rotation_angle)  # 转换为普通字典
                self._fixed_rotation_angles = {
                    "x": float(rotation_angle.get("x", 0.0)),
                    "y": float(rotation_angle.get("y", 0.0)),
                    "z": float(rotation_angle.get("z", 0.0))
                }
            else:
                raise ValueError(f"Invalid rotation_angle type: {type(rotation_angle)}")
        
        # 解析随机旋转范围参数
        self._rotation_ranges = None
        if random_rotation:
            # 绝对值范围优先：例如 abs_rotation_range_deg: [55, 90]
            if abs_rotation_range_deg is not None:
                arr = abs_rotation_range_deg
                if isinstance(arr, (list, tuple, ListConfig)):
                    arr = list(arr)
                if isinstance(arr, (dict, DictConfig)):
                    arr = dict(arr)

                def _norm_pair(v, default):
                    if v is None:
                        lo, hi = default
                    else:
                        if isinstance(v, (list, tuple, ListConfig)):
                            v = list(v)
                        if not (isinstance(v, list) and len(v) == 2):
                            raise ValueError("abs_rotation_range_deg per-axis value must be [min_abs, max_abs]")
                        lo, hi = float(v[0]), float(v[1])
                    if lo < 0 or hi < 0:
                        raise ValueError("abs_rotation_range_deg must be non-negative")
                    if hi < lo:
                        lo, hi = hi, lo
                    return [lo, hi]

                if isinstance(arr, dict):
                    self._abs_rotation_ranges = {
                        "x": _norm_pair(arr.get("x", None), [0.0, 0.0]),
                        "y": _norm_pair(arr.get("y", None), [0.0, 0.0]),
                        "z": _norm_pair(arr.get("z", None), [0.0, 0.0]),
                    }
                elif isinstance(arr, list) and len(arr) == 2 and all(isinstance(x, (int, float)) for x in arr):
                    lo, hi = float(arr[0]), float(arr[1])
                    if lo < 0 or hi < 0:
                        raise ValueError("abs_rotation_range_deg must be non-negative")
                    if hi < lo:
                        lo, hi = hi, lo
                    self._abs_rotation_ranges = {"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}
                elif isinstance(arr, list) and len(arr) == 3:
                    # [[x_lo,x_hi],[y_lo,y_hi],[z_lo,z_hi]]
                    self._abs_rotation_ranges = {
                        "x": _norm_pair(arr[0], [0.0, 0.0]),
                        "y": _norm_pair(arr[1], [0.0, 0.0]),
                        "z": _norm_pair(arr[2], [0.0, 0.0]),
                    }
                else:
                    raise ValueError(
                        "Invalid abs_rotation_range_deg. Use [min,max] or [[x_min,x_max],[y_min,y_max],[z_min,z_max]] or {x:[min,max],y:[min,max],z:[min,max]}"
                    )

            if rotation_range is None:
                # 默认：Z轴 [-30, 30] 度
                self._rotation_ranges = {"x": [-30.0, 30.0], "y": [-30.0, 30.0], "z": [-30.0, 30.0]}
            elif isinstance(rotation_range, (int, float)):
                # 单个数字：Z轴范围，例如 30.0 表示 [-30, 30]
                val = float(rotation_range)
                self._rotation_ranges = {"x": [-val, val], "y": [-val, val], "z": [-val, val]}
            elif isinstance(rotation_range, (list, tuple, np.ndarray, ListConfig)):
                # 列表：[[x_min, x_max], [y_min, y_max], [z_min, z_max]] 或 [z_min, z_max]
                rotation_range = list(rotation_range)  # 转换为普通列表
                if len(rotation_range) == 2 and all(isinstance(x, (int, float)) for x in rotation_range):
                    # [z_min, z_max] 格式
                    self._rotation_ranges = {
                        "x": [-30.0, 30.0],
                        "y": [-30.0, 30.0],
                        "z": [float(rotation_range[0]), float(rotation_range[1])]
                    }
                elif len(rotation_range) == 3:
                    # [[x_min, x_max], [y_min, y_max], [z_min, z_max]] 格式
                    # 处理嵌套的 ListConfig
                    x_range = list(rotation_range[0]) if isinstance(rotation_range[0], (list, tuple, ListConfig)) else rotation_range[0]
                    y_range = list(rotation_range[1]) if isinstance(rotation_range[1], (list, tuple, ListConfig)) else rotation_range[1]
                    z_range = list(rotation_range[2]) if isinstance(rotation_range[2], (list, tuple, ListConfig)) else rotation_range[2]
                    self._rotation_ranges = {
                        "x": [float(x_range[0]), float(x_range[1])],
                        "y": [float(y_range[0]), float(y_range[1])],
                        "z": [float(z_range[0]), float(z_range[1])]
                    }
                else:
                    raise ValueError("rotation_range list must have 2 or 3 elements")
            elif isinstance(rotation_range, (dict, DictConfig)):
                # 字典：{"x": [min, max], "y": [min, max], "z": [min, max]}
                rotation_range = dict(rotation_range)  # 转换为普通字典
                def get_range(key, default):
                    val = rotation_range.get(key, default)
                    if isinstance(val, (list, tuple, ListConfig)):
                        val = list(val)
                    return [float(val[0]), float(val[1])]
                self._rotation_ranges = {
                    "x": get_range("x", [-30.0, 30.0]),
                    "y": get_range("y", [-30.0, 30.0]),
                    "z": get_range("z", [-30.0, 30.0])
                }
            else:
                raise ValueError(f"Invalid rotation_range type: {type(rotation_range)}")
        
        # 当前旋转角度（在reset时生成）
        self._rotation_angles = {"x": 0.0, "y": 0.0, "z": 0.0}
        return

    def _sample_with_min_abs(self, lo: float, hi: float, min_abs: float) -> float:
        """
        从区间 [lo, hi] 采样，要求 |x| >= min_abs（若可能）。
        若连续尝试未满足，则退化为选择最靠近且满足阈值的边界值。
        """
        if min_abs <= 0:
            return float(np.random.uniform(lo, hi))

        # 若区间内不存在满足 |x| >= min_abs 的值，则直接返回边界中绝对值更大的那个
        if max(abs(lo), abs(hi)) < min_abs:
            return float(lo if abs(lo) >= abs(hi) else hi)

        for _ in range(50):
            v = float(np.random.uniform(lo, hi))
            if abs(v) >= min_abs:
                return v

        # 兜底：选择能满足阈值的最近边界
        candidates = []
        if abs(lo) >= min_abs:
            candidates.append(lo)
        if abs(hi) >= min_abs:
            candidates.append(hi)
        if candidates:
            return float(candidates[np.argmax(np.abs(candidates))])
        # 理论上不会到这里（已在上面返回），保险返回 hi
        return float(hi)

    def _apply_rotation(self, end_effector_orientation: np.ndarray) -> np.ndarray:
        """
        对末端执行器方向应用旋转。
        
        Args:
            end_effector_orientation: 原始方向（四元数）
            
        Returns:
            旋转后的方向（四元数）
        """
        # 将四元数转换为旋转对象
        r = R.from_quat(end_effector_orientation)
        
        # 应用绕X、Y、Z轴的旋转（使用欧拉角 'xyz' 顺序）
        rotation_euler = R.from_euler('xyz', [
            np.radians(self._rotation_angles["x"]),
            np.radians(self._rotation_angles["y"]),
            np.radians(self._rotation_angles["z"])
        ])
        
        # 应用旋转
        r_rotated = rotation_euler * r
        
        # 转换回四元数
        return r_rotated.as_quat()

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
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self.is_done():
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        # 应用旋转，使放置角度异常
        rotated_orientation = self._apply_rotation(end_effector_orientation)
        # 标记异常（首次进入放置阶段即触发）
        self._exception_fired = True
        
        if self._event == 0:
            self.target_position = place_position.copy()
            self.target_position[2] += pre_place_z / get_stage_units()
            print(f"放置阶段0：应用旋转角度 X={self._rotation_angles['x']:.1f}°, Y={self._rotation_angles['y']:.1f}°, Z={self._rotation_angles['z']:.1f}°")
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=rotated_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
                    self._event += 1
                    self._t = 0
        elif self._event == 1:
            self.target_position = place_position.copy()
            self.target_position[2] += place_offset_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=rotated_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position - gripper_position)
                if xy_distance < 0.02:
                    self._event += 1
                    self._t = 0
        elif self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="open")
            self.target_position = place_position.copy()
            self.target_position[2] += 0.15 / get_stage_units()
            self.target_position[0] -= 0.15 / get_stage_units()
            gripper_control.release_object()
        elif self._event == 4:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=rotated_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
                    self._event += 1
                    self._t = 0
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        return target_joint_positions

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
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
        
        # 生成随机旋转角度（如果启用随机模式）
        if self._random_rotation:
            if self._abs_rotation_ranges is not None:
                def _sample_abs(lo: float, hi: float) -> float:
                    # |angle| in [lo, hi], random sign
                    mag = float(np.random.uniform(lo, hi)) if hi > lo else float(lo)
                    if mag == 0.0:
                        return 0.0
                    sign = -1.0 if np.random.rand() < 0.5 else 1.0
                    return float(sign * mag)

                self._rotation_angles = {
                    "x": _sample_abs(self._abs_rotation_ranges["x"][0], self._abs_rotation_ranges["x"][1]),
                    "y": _sample_abs(self._abs_rotation_ranges["y"][0], self._abs_rotation_ranges["y"][1]),
                    "z": _sample_abs(self._abs_rotation_ranges["z"][0], self._abs_rotation_ranges["z"][1]),
                }
                print(
                    f"生成随机旋转角度(绝对值范围): "
                    f"X={self._rotation_angles['x']:.1f}°, "
                    f"Y={self._rotation_angles['y']:.1f}°, "
                    f"Z={self._rotation_angles['z']:.1f}°"
                )
            else:
                self._rotation_angles = {
                    "x": self._sample_with_min_abs(self._rotation_ranges["x"][0], self._rotation_ranges["x"][1], self._min_abs_rotation_deg),
                    "y": self._sample_with_min_abs(self._rotation_ranges["y"][0], self._rotation_ranges["y"][1], self._min_abs_rotation_deg),
                    "z": self._sample_with_min_abs(self._rotation_ranges["z"][0], self._rotation_ranges["z"][1], self._min_abs_rotation_deg),
                }
                print(
                    f"生成随机旋转角度: "
                    f"X={self._rotation_angles['x']:.1f}°, "
                    f"Y={self._rotation_angles['y']:.1f}°, "
                    f"Z={self._rotation_angles['z']:.1f}° (min_abs={self._min_abs_rotation_deg:.1f}°)"
                )
        else:
            # 使用固定角度
            self._rotation_angles = self._fixed_rotation_angles.copy()
        return

    def get_wrong_angle_info(self):
        """返回一次性放置角度异常信息，用于截图/命名。"""
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            info = {
                "type": "place_wrong_angle",
                "message": (
                    f"Place angle deviation: "
                    f"X={self._rotation_angles['x']:.1f}°, "
                    f"Y={self._rotation_angles['y']:.1f}°, "
                    f"Z={self._rotation_angles['z']:.1f}°"
                ),
                "suffix": "erro",
                "color": (0, 0, 255),
            }
            return info
        return None

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

