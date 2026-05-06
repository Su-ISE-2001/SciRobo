import re
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    FINISHED = "finished"

class PlaceTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.current_phase = Phase.PICKING
        # Failure mode: "fail_grasp", "fail_position", "drop" or None
        self.failure_mode = cfg.task.get("failure_mode", None)
        self._frame_exception = None

    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        
        failure_mode = cfg.task.get("failure_mode", None)
        
        # 根据 failure_mode 选择不同的 pick 控制器
        # 注意：drop/wrong_angle/fail_position 等只在 place 阶段生效，pick 阶段正常执行
        if failure_mode == "fail_grasp":
            from controllers.atomic_actions.pick_controller_fail_grasp import PickControllerFailGrasp
            position_offset = cfg.task.get("position_offset", 0.05)  # 默认偏移0.05米
            self.pick_controller = PickControllerFailGrasp(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02],
                position_offset=position_offset
            )
        elif failure_mode == "fail_angle_pick":
            from controllers.atomic_actions.pick_controller_fail_angle import PickControllerFailAngle
            angle_offset_deg = cfg.task.get("angle_offset_deg", 45.0)
            deviation_report_threshold_deg = cfg.task.get("angle_report_threshold_deg", 20.0)
            angle_offset_range_deg = cfg.task.get("angle_offset_range_deg", None)
            angle_offset_choices_deg = cfg.task.get("angle_offset_choices_deg", None)
            self.pick_controller = PickControllerFailAngle(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02],
                angle_offset_deg=angle_offset_deg,
                angle_offset_range_deg=angle_offset_range_deg,
                angle_offset_choices_deg=angle_offset_choices_deg,
                deviation_report_threshold_deg=deviation_report_threshold_deg,
            )
        elif failure_mode == "fail_position_pick":
            from controllers.atomic_actions.pick_controller_fail_position import PickControllerFailPosition
            position_offset = cfg.task.get("position_offset", 0.05)
            self.pick_controller = PickControllerFailPosition(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02],
                position_offset=position_offset,
            )
        elif failure_mode == "incomplete_close_pick":
            from controllers.atomic_actions.pick_controller_incomplete_close import PickControllerIncompleteClose
            incomplete_close_scale = cfg.task.get("incomplete_close_scale", 1.5)
            extra_gap = cfg.task.get("extra_gap", 0.0)
            gap_report_threshold = cfg.task.get("gap_report_threshold", 0.002)
            self.pick_controller = PickControllerIncompleteClose(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02],
                incomplete_close_scale=incomplete_close_scale,
                extra_gap=extra_gap,
                gap_report_threshold=gap_report_threshold,
            )
        else:
            # drop 模式下，pick 阶段正常执行（不在这里处理 drop）
            self.pick_controller = PickController(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
            )
        
        # 根据 failure_mode 选择不同的 place 控制器
        if failure_mode == "fail_position":
            from controllers.atomic_actions.place_controller_fail_position import PlaceControllerFailPosition
            position_offset = cfg.task.get("position_offset", 0.08)  # 默认偏移0.08米
            self.place_controller = PlaceControllerFailPosition(
                name="place_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                position_offset=position_offset
            )
        elif failure_mode == "drop":
            from controllers.atomic_actions.place_controller_drop import PlaceControllerDrop
            self.place_controller = PlaceControllerDrop(
                name="place_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
            )
        elif failure_mode == "wrong_angle":
            from controllers.atomic_actions.place_controller_wrong_angle import PlaceControllerWrongAngle
            rotation_angle = cfg.task.get("rotation_angle", None)
            random_rotation = cfg.task.get("random_rotation", True)  # 默认使用随机旋转
            rotation_range = cfg.task.get("rotation_range", None)  # 随机旋转范围
            min_abs_rotation_deg = cfg.task.get("min_abs_rotation_deg", 0.0)
            abs_rotation_range_deg = cfg.task.get("abs_rotation_range_deg", None)
            # 支持多种格式：
            # - rotation_angle: 固定角度（当 random_rotation=False 时使用）
            #   - 单个数字：绕Z轴旋转
            #   - 列表 [x, y, z]：绕X、Y、Z轴旋转
            #   - 字典 {"x": 30, "y": 20, "z": 10}：分别指定各轴旋转角度
            # - rotation_range: 随机旋转范围（当 random_rotation=True 时使用）
            #   - 单个数字：Z轴范围，例如 30.0 表示 [-30, 30]
            #   - 列表 [[x_min, x_max], [y_min, y_max], [z_min, z_max]] 或 [z_min, z_max]
            #   - 字典 {"x": [min, max], "y": [min, max], "z": [min, max]}
            self.place_controller = PlaceControllerWrongAngle(
                name="place_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                rotation_angle=rotation_angle,
                random_rotation=random_rotation,
                rotation_range=rotation_range,
                min_abs_rotation_deg=min_abs_rotation_deg,
                abs_rotation_range_deg=abs_rotation_range_deg,
            )
        else:
            self.place_controller = PlaceController(
                name="place_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
            )
        
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot):
        """Initialize controller for inference mode."""
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        super()._init_infer_mode(cfg, robot)

    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self._frame_exception = None
        self.pick_controller.reset()
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.place_controller.reset()
        else:
            self.inference_engine.reset()

    def _check_phase_success(self):
        """Check if current phase is successful based on object position."""
        object_pos = self.state['object_position']
        target_position = self.state['target_position']
        
        if self.current_phase == Phase.PICKING:
            return object_pos[2] > self.initial_position[2] + 0.1
        elif self.current_phase == Phase.PLACING:
            success = (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.05 and abs(object_pos[2] - self.initial_position[2]) < 0.05)
            return success


    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        if self.initial_position is None:
            self.initial_position = self.state['object_position']
        if self.initial_size is None:
            self.initial_size = self.state['object_size']
            
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Execute collection mode step."""
        success = self._check_phase_success()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                )
                self._maybe_mark_pick_exception(state)
                # 缓存步骤数据（pick阶段也需要）
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
            else:
                action = self.place_controller.forward(
                    place_position = state['target_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    gripper_position=state['gripper_position']
                )
                self._maybe_mark_place_exception(state)
                # 对于 drop 模式，只有在烧杯还没有掉落时才更新位置
                if self.failure_mode == "drop":
                    if self.gripper_control.grasped_object_path is not None:
                        self.gripper_control.update_grasped_object_position()
                else:
                    # 非 drop 模式，正常更新位置
                    self.gripper_control.update_grasped_object_position()
                # 缓存步骤数据（place阶段）
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1],
                        language_instruction=self.get_language_instruction()
                    )
            
            return action, False, False

        # 处理阶段转换和任务完成
        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to place...")
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False
            elif self.current_phase == Phase.PLACING:
                print("Place task success!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
        else:
            # 任务失败，保存失败数据而不是清除缓存
            if self.failure_mode == "drop" and self.current_phase == Phase.PLACING:
                print(f"{self.current_phase.value} task failed（烧杯在移动过程中滑落）！保存失败数据...")
            else:
                print(f"{self.current_phase.value} task failed! 保存失败数据...")
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self._last_success = False
            self.current_phase = Phase.FINISHED
            return None, True, False
        
        return None, False, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        if self.current_phase == Phase.PICKING:
            action = None
            action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                )    
        else:
            state['language_instruction'] = self.get_language_instruction()
            action = self.inference_engine.step_inference(state)
            self._maybe_mark_place_exception(state)
        return action, False, self.is_success()

    def is_success(self):
        object_pos = self.state["object_position"]
        target_position = self.state['target_position']
        if (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.05 and abs(object_pos[2] - self.initial_position[2]) < 0.05):
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return True
        return False

    def get_language_instruction(self) -> str:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ',' ').lower()
        self._language_instruction = f"Pick up the {object_name} from the table and place it at the target"
        return self._language_instruction

    def _maybe_mark_pick_exception(self, state):
        """记录夹取阶段相关异常用于截图标注/命名。"""
        info = None
        if self.failure_mode == "fail_grasp" and hasattr(self.pick_controller, "get_grasp_error_info"):
            info = self.pick_controller.get_grasp_error_info()
        elif self.failure_mode == "fail_angle_pick" and hasattr(self.pick_controller, "get_angle_deviation_info"):
            info = self.pick_controller.get_angle_deviation_info()
        elif self.failure_mode == "fail_position_pick" and hasattr(self.pick_controller, "get_position_error_info"):
            info = self.pick_controller.get_position_error_info()
        elif self.failure_mode == "incomplete_close_pick" and hasattr(self.pick_controller, "get_incomplete_close_info"):
            info = self.pick_controller.get_incomplete_close_info()

        if info is None:
            return
        info["frame_idx"] = state.get("frame_idx")
        info["episode"] = self._episode_num
        self._frame_exception = info

    def _maybe_mark_place_exception(self, state):
        """记录放置阶段相关异常用于截图标注/命名。"""
        info = None
        if self.failure_mode == "drop" and hasattr(self.place_controller, "get_drop_info"):
            info = self.place_controller.get_drop_info()
        elif self.failure_mode == "wrong_angle" and hasattr(self.place_controller, "get_wrong_angle_info"):
            info = self.place_controller.get_wrong_angle_info()
        elif self.failure_mode == "fail_position" and hasattr(self.place_controller, "get_place_position_error_info"):
            info = self.place_controller.get_place_position_error_info()

        if info is None:
            return
        info["frame_idx"] = state.get("frame_idx")
        info["episode"] = self._episode_num
        self._frame_exception = info

    def consume_frame_exception(self):
        """供外部读取并清空当前帧异常标记。"""
        info = self._frame_exception
        self._frame_exception = None
        return info