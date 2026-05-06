from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from utils.object_utils import ObjectUtils

class PickControllerDrop(BaseController):
    """
    夹取控制器，模拟夹取过程中物体掉落的情况。
    机器人在成功夹取物体后，在抬起过程中会释放物体，导致物体掉落。
    """
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
        drop_phase: int = 5  # 在哪个阶段掉落（5是抬起阶段）
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0
        self.drop_phase = drop_phase  # 掉落阶段

        if events_dt is None:
            self._events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.05, 0.004, 0.006]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 8:
                raise Exception(f"events_dt length must be 8, got {len(self._events_dt)}")

        self._cspace_controller = cspace_controller
        self._start = True
        self.object_size = None
        self._position_threshold = position_threshold
        self._robot_position = None
        self._exception_fired = False
        self._exception_reported = False

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
        gripper_distances: float = None
    ) -> ArticulationAction:
        """计算当前夹取阶段的关节位置，在抬起过程中释放物体。

        Args:
            picking_position (np.ndarray): 目标夹取位置。
            current_joint_positions (np.ndarray): 机器人当前关节位置。
            object_name (str): 要夹取的物体名称。
            object_size (np.ndarray): 物体尺寸。
            gripper_control: 夹爪控制器实例。
            gripper_position (np.ndarray): 夹爪当前位置。
            end_effector_orientation (np.ndarray, optional): 末端执行器目标姿态。默认为 [0, pi, 0] 欧拉角。
            pre_offset_z (float): 预夹取Z轴偏移。
            after_offset_z (float): 夹取后Z轴偏移。
            pre_offset_x (float): 预夹取X轴偏移。
            gripper_distances (float): 夹爪距离。

        Returns:
            ArticulationAction: 机器人要执行的关节位置。
        """
        # 验证输入参数
        if picking_position is None:
            raise ValueError("picking_position cannot be None")
        picking_position = np.array(picking_position).copy()
        
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
            gripper_distances
        )

        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0
            
        return target_joint_positions

    def get_drop_info(self):
        """返回一次性掉落异常信息用于命名/标注。"""
        if self._exception_fired and not self._exception_reported:
            self._exception_reported = True
            return {
                "type": "drop",
                "message": "Object dropped intentionally during lift",
                "suffix": "erro",
                "color": (0, 0, 255),
            }
        return None

    def _handle_start_state(self, current_joint_positions):
        """处理初始状态，打开夹爪。

        Args:
            current_joint_positions (np.ndarray): 机器人当前关节位置。

        Returns:
            ArticulationAction: 打开夹爪的关节位置。
        """
        self._start = False
        target_joint_positions = [None] * current_joint_positions.shape[0]
        target_joint_positions[7] = 0.04 / get_stage_units()
        target_joint_positions[8] = 0.04 / get_stage_units()
        return ArticulationAction(joint_positions=target_joint_positions)

    def _execute_phase(self, picking_position, end_effector_orientation, current_joint_positions, object_name, gripper_control, gripper_position, gripper_distances):
        """执行当前阶段的夹取序列，在抬起阶段释放物体。

        Args:
            picking_position (np.ndarray): 目标夹取位置。
            end_effector_orientation (np.ndarray): 末端执行器目标姿态。
            current_joint_positions (np.ndarray): 机器人当前关节位置。
            object_name (str): 目标物体名称。
            gripper_control: 夹爪控制器实例。
            gripper_position (np.ndarray): 夹爪当前位置。
            gripper_distances (float): 夹爪距离。

        Returns:
            ArticulationAction: 机器人控制的目标关节位置。
        """
        
        approach_dir = self._calculate_approach_direction(picking_position)
        
        if self._event == 0:
            # 阶段0：移动到物体上方（正常位置）
            picking_position = picking_position + approach_dir * (self.pre_offset_x / get_stage_units())
            picking_position[2] += self.object_size[2] + self.pre_offset_z
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 1:
            # 阶段1：降低末端执行器接近物体（正常位置）
            picking_position = picking_position + approach_dir * (0.1 / get_stage_units())
            picking_position[2] += self.get_pickprez_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 2:
            # 阶段2：定位末端执行器进行抓取（正常位置）
            picking_position[2] += self.get_pickz_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            z_distance = abs(gripper_position[2] - picking_position[2])
            if xy_distance < 0.005 and z_distance < 0.005:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 3:
            # 阶段3：等待机器人动力学稳定
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 4:
            # 阶段4：关闭夹爪抓取物体（正常抓取）
            target_joint_positions = [None] * current_joint_positions.shape[0]
            if gripper_distances is None:
                gripper_distances = self.get_gripper_distance(object_name) / get_stage_units()
            target_joint_positions[7] = gripper_distances
            target_joint_positions[8] = gripper_distances
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            # 设置中间抬起位置（抬起一段距离，但保持夹爪关闭）
            self.intermediate_position = picking_position.copy()
            self.intermediate_position[2] += (self.after_offset_z * 0.4) / get_stage_units()  # 抬起40%的高度
            # 设置最终抬起位置
            self.target_position = picking_position.copy()
            self.target_position[2] += self.after_offset_z / get_stage_units()
            # 正常添加物体到夹爪（与正常 pick_controller 相同的方式）
            if "glass" in object_name:
                gripper_control.add_object_to_gripper("/World/glass_rod/Cylinder", "/World/Franka/panda_hand/tool_center")
            # 对于其他物体，gripper_control 会在适当的时候自动处理
            return target_joint_positions

        elif self._event == 5:
            # 阶段5：抬起一段距离（保持夹爪关闭）
            arm_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.intermediate_position,
                target_end_effector_orientation=end_effector_orientation
            )
            # 创建包含所有关节的数组（包括夹爪关节）
            target_joint_positions = [None] * current_joint_positions.shape[0]
            # 复制机械臂关节位置
            if arm_joint_positions.joint_positions is not None:
                for i in range(len(arm_joint_positions.joint_positions)):
                    if arm_joint_positions.joint_positions[i] is not None:
                        target_joint_positions[i] = arm_joint_positions.joint_positions[i]
            # 保持夹爪关闭（使用抓取距离）
            if gripper_distances is None:
                gripper_distances = self.get_gripper_distance(object_name) / get_stage_units()
            target_joint_positions[7] = gripper_distances
            target_joint_positions[8] = gripper_distances
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            
            xy_distance = np.linalg.norm(gripper_position[:2] - self.intermediate_position[:2])
            z_distance = abs(gripper_position[2] - self.intermediate_position[2])
            if xy_distance < self._position_threshold and z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 6:
            # 阶段6：释放物体并打开夹爪，继续抬起（导致物体掉落）
            # 释放物体
            gripper_control.release_object()
            # 标记掉落异常
            self._exception_fired = True
            # 抬起动作
            arm_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            # 创建包含所有关节的数组（包括夹爪关节）
            target_joint_positions = [None] * current_joint_positions.shape[0]
            # 复制机械臂关节位置
            if arm_joint_positions.joint_positions is not None:
                for i in range(len(arm_joint_positions.joint_positions)):
                    if arm_joint_positions.joint_positions[i] is not None:
                        target_joint_positions[i] = arm_joint_positions.joint_positions[i]
            # 打开夹爪（导致物体掉落）
            target_joint_positions[7] = 0.04 / get_stage_units()
            target_joint_positions[8] = 0.04 / get_stage_units()
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            
            xy_distance = np.linalg.norm(gripper_position[:2] - self.target_position[:2])
            z_distance = abs(gripper_position[2] - self.target_position[2])
            if xy_distance < self._position_threshold and z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions
        else:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """重置控制器到初始阶段。

        Args:
            events_dt (List[float], optional): 新的阶段持续时间。默认为 None。

        Raises:
            Exception: 如果 events_dt 不是列表或 numpy 数组，或其长度不是 8。
        """
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
            if len(self._events_dt) != 8:
                raise Exception(f"events_dt length must be 8, got {len(self._events_dt)}")

        self._start = True
        self.object_size = None
        self._robot_position = None
        self._exception_fired = False
        self._exception_reported = False

    def is_done(self) -> bool:
        """检查夹取序列是否完成。

        Returns:
            bool: 如果达到最终阶段则返回 True，否则返回 False。
        """
        return self._event >= len(self._events_dt)

    def get_gripper_distance(self, item_name):
        """确定指定物体的夹爪开口距离。

        Args:
            item_name (str): 要抓取的物体名称。

        Returns:
            float: 夹爪手指距离（米）。
        """
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
            "BalaoVolumetrico_100mL":0.002,
            "xform": 0.003,
            "titrationflasks": 0.020,
            "titrationflasks_01": 0.003,
        }

        for key in gripper_distances:
            if key == item_name.lower():
                return gripper_distances[key]

        return 0.01

    def get_pickz_offset(self, item_name):
        """计算最终抓取位置的垂直偏移。

        Args:
            item_name (str): 要抓取的物体名称。

        Returns:
            float: 垂直偏移（米）。
        """
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
        """计算预抓取位置的垂直偏移。

        Args:
            item_name (str): 要抓取的物体名称。

        Returns:
            float: 垂直偏移（米）。
        """
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
