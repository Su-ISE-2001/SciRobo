import re
from typing import Optional
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum
from utils.task_utils import TaskUtils

from .atomic_actions.pick_controller import PickController
from .atomic_actions.pour_controller import PourController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    POURING = "pouring"
    FINISHED = "finished"

class PourTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.task_utils = TaskUtils.get_instance()
        self.initial_quaternion = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.last_error_info = None
        self.current_phase = Phase.PICKING
        # Failure mode: "fail_position", "fail_position_pick", "fail_angle_pick", "incomplete_close_pick", "drop", "incomplete_angle" or None
        self.failure_mode = cfg.task.get("failure_mode", None)
        self._frame_exception = None
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        """Initialize controller for data collection mode."""
        
        # Use different controllers based on failure mode
        failure_mode = cfg.task.get("failure_mode", None)
        
        # 根据 failure_mode 选择不同的 pick 控制器
        if failure_mode == "fail_position_pick":
            from controllers.atomic_actions.pick_controller_fail_position import PickControllerFailPosition
            position_offset = cfg.task.get("position_offset", 0.03)  # 默认偏移0.03米
            self.pick_controller = PickControllerFailPosition(
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
                deviation_report_threshold_deg=deviation_report_threshold_deg,
                angle_offset_range_deg=angle_offset_range_deg,
                angle_offset_choices_deg=angle_offset_choices_deg
            )
        elif failure_mode == "incomplete_close_pick":
            from controllers.atomic_actions.pick_controller_incomplete_close import PickControllerIncompleteClose
            incomplete_close_scale = cfg.task.get("incomplete_close_scale", 1.6)
            extra_gap = cfg.task.get("extra_gap", 0.0)
            gap_report_threshold = cfg.task.get("gap_report_threshold", 0.002)
            self.pick_controller = PickControllerIncompleteClose(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02],
                incomplete_close_scale=incomplete_close_scale,
                extra_gap=extra_gap,
                gap_report_threshold=gap_report_threshold
            )
        else:
            self.pick_controller = PickController(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
            )
        if failure_mode == "fail_position":
            from controllers.atomic_actions.pour_controller_fail_position import PourControllerFailPosition
            position_offset = cfg.task.get("position_offset", 0.03)  # 默认偏移0.03米
            self.pour_controller = PourControllerFailPosition(
                name="pour_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.006, 0.002, 0.009, 0.01, 0.009, 0.01],
                position_offset=position_offset
            )
        elif failure_mode == "drop":
            from controllers.atomic_actions.pour_controller_drop import PourControllerDrop
            self.pour_controller = PourControllerDrop(
                name="pour_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.006, 0.002, 0.009, 0.01, 0.009, 0.01]
            )
        elif failure_mode == "incomplete_angle":
            from controllers.atomic_actions.pour_controller_incomplete_angle import PourControllerIncompleteAngle
            angle_reduction = cfg.task.get("angle_reduction", 0.5)  # 默认减少50%
            self.pour_controller = PourControllerIncompleteAngle(
                name="pour_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.006, 0.002, 0.009, 0.01, 0.009, 0.01],
                angle_reduction=angle_reduction
            )
        else:
            self.pour_controller = PourController(
                name="pour_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.006, 0.002, 0.009, 0.01, 0.009, 0.01]
            )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot=None):
        super()._init_infer_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )

    def reset(self):
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.initial_quaternion = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.last_error_info = None
        self._frame_exception = None
        self.pick_controller.reset()
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pour_controller.reset()
        else:
            self.inference_engine.reset()

    def _check_phase_success(self):
        """Check if current phase is successful."""
        object_pos = self.state['object_position']
        self.last_error_info = None 
        
        if self.initial_position is None:
            raise ValueError("initial_position not set")

        if self.current_phase == Phase.PICKING:
            # 在 fail_position 模式下，只要 pick_controller 完成了就认为 pick 成功
            # 这样确保能进入 pour 阶段，在 pour 阶段因为位置错误而失败
            if self.failure_mode == "fail_position" and self.pick_controller.is_done():
                return True
            
            # 在 fail_position_pick、fail_angle_pick 或 incomplete_close_pick 模式下，如果 pick 控制器完成了，也允许进入 pour 阶段
            # 但由于位置/角度/闭合不足错误，夹取可能失败，导致物体掉落或无法成功夹取
            if self.failure_mode in ["fail_position_pick", "fail_angle_pick", "incomplete_close_pick"] and self.pick_controller.is_done():
                return True
            
            required_height = self.initial_position[2] + 0.12
            success = object_pos[2] > required_height
            if not success:
                self.last_error_info = {
                    'phase': 'PICKING',
                    'current_height': object_pos[2],
                    'required_height': required_height,
                    'height_diff': object_pos[2] - required_height
                }
            return success
            
        elif self.current_phase == Phase.POURING:
            if self.initial_quaternion is None:
                self.initial_quaternion = self.state['object_quaternion']
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Initial quaternion not set yet'
                }
                return False
                
            current_quat = self.state['object_quaternion']
            
            # First check if we're close enough to target for pouring
            xy_dist = np.linalg.norm(object_pos[:2] - self.state['target_position'][:2])
            pour_threshold = self.task_utils.get_pour_threshold(self.state['object_name'], self.state['object_size']) + 0.05
            
            if xy_dist > pour_threshold:
                self.last_error_info = {
                    'phase': 'POURING',
                    'current_distance': xy_dist,
                    'required_distance': pour_threshold,
                    'distance_diff': xy_dist - pour_threshold
                }
                return False
            
            if not self.pour_complete:
                # print(self.initial_quaternion, current_quat)
                self.pour_complete = self.task_utils.check_rotation_angle(
                    self.initial_quaternion, 
                    current_quat,
                    threshold_degrees=50
                )
                if not self.pour_complete:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Pour rotation not complete yet',
                        'pour_complete': self.pour_complete
                    }
                return False
                
            # After pour complete, check if returned to original orientation
            if not self.return_complete:
                rotation_diff = self.task_utils.check_rotation_angle(
                    self.initial_quaternion,
                    current_quat,
                    threshold_degrees=30  # smaller threshold for return position
                )
                if not rotation_diff:
                    self.return_complete = True
                    self.return_timer = 0
                else:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Return rotation not complete yet',
                        'return_complete': self.return_complete
                    }
                return False
                
            # Wait for 2 seconds in return position
            if self.return_complete and object_pos[2] > self.initial_position[2] + 0.05:
                self.return_timer += 0.012
                success = self.return_timer >= 1.0
                if not success:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Waiting for return timer',
                        'return_timer': self.return_timer,
                        'required_time': 1.0
                    }
                return success
            else:
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Object not in correct position for return timer',
                    'current_height': object_pos[2],
                    'required_height': self.initial_position[2] + 0.05,
                    'return_complete': self.return_complete
                }
                return False
        
        return False
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
        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to pour...")
                self.current_phase = Phase.POURING
                self.active_controller = self.pour_controller
                return None, False, False
            elif self.current_phase == Phase.POURING:
                print("Pour task success!")
                # 在写入数据前检查是否已达到最大 episode 数
                if self.data_collector.episode_count >= self.data_collector.max_episodes:
                    print(f"Reached max_episodes ({self.data_collector.max_episodes}), stopping data collection.")
                    self._last_success = False
                    self.current_phase = Phase.FINISHED
                    return None, True, False
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
        
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        # 在 fail_position、fail_position_pick、fail_angle_pick 或 incomplete_close_pick 模式下，如果 pick 控制器完成了，直接切换到 pour 阶段
        if (self.current_phase == Phase.PICKING and 
            self.failure_mode in ["fail_position", "fail_position_pick", "fail_angle_pick", "incomplete_close_pick"] and 
            self.pick_controller.is_done()):
            print(f"Pick controller done ({self.failure_mode} mode)! Switching to pour...")
            self.current_phase = Phase.POURING
            self.active_controller = self.pour_controller
            return None, False, False

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
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    pre_offset_x=0.05,
                    pre_offset_z=0.05,
                    after_offset_z=0.3
                )
                # 检查并标记夹取位置错误异常
                self._maybe_mark_pick_exception(state)
            else:
                # 对于 drop 模式，需要传递 gripper_control 参数
                if self.failure_mode == "drop":
                    action = self.pour_controller.forward(
                        articulation_controller=self.robot.get_articulation_controller(),
                        source_size=self.initial_size,
                        target_position=state['target_position'],
                        current_joint_velocities=self.robot.get_joint_velocities(),
                        gripper_control=self.gripper_control,
                        pour_speed=-1,
                        source_name=state['object_name'],
                        gripper_position=state['gripper_position'],
                    )
                    # 只有在试管还没有掉落时才更新位置
                    if self.gripper_control.grasped_object_path is not None:
                        self.gripper_control.update_grasped_object_position()
                    # 检查并标记掉落异常
                    self._maybe_mark_pour_exception(state)
                else:
                    action = self.pour_controller.forward(
                        articulation_controller=self.robot.get_articulation_controller(),
                        source_size=self.initial_size,
                        target_position=state['target_position'],
                        current_joint_velocities=self.robot.get_joint_velocities(),
                        pour_speed=-1,
                        source_name=state['object_name'],
                        gripper_position=state['gripper_position'],
                    )
                    self.gripper_control.update_grasped_object_position()
                    # 检查并标记倾倒阶段异常（如 fail_position、incomplete_angle）
                    self._maybe_mark_pour_exception(state)
            
            # 缓存步骤数据（两个阶段都需要）
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False

        print(f"{self.current_phase.value} task failed!")
        if self.last_error_info is not None:
            print(f"Phase failure details: {self.last_error_info}")
        
        # 在写入数据前检查是否已达到最大 episode 数
        if self.data_collector.episode_count >= self.data_collector.max_episodes:
            print(f"Reached max_episodes ({self.data_collector.max_episodes}), stopping data collection.")
            self._last_success = False
            self.current_phase = Phase.FINISHED
            return None, True, False
        
        # 保存失败数据，而不是清除缓存
        if self.failure_mode == "drop" and self.current_phase == Phase.POURING:
            print("Pour task failed（试管在移动过程中掉落）！保存失败数据...")
        elif self.failure_mode == "incomplete_angle" and self.current_phase == Phase.POURING:
            print("Pour task failed（倾倒角度不当）！保存失败数据...")
        elif self.failure_mode == "fail_position_pick" and self.current_phase == Phase.PICKING:
            print("Pour task failed（夹取位置错误）！保存失败数据...")
        elif self.failure_mode == "fail_angle_pick" and self.current_phase == Phase.PICKING:
            print("Pour task failed（夹取角度错误）！保存失败数据...")
        elif self.failure_mode == "incomplete_close_pick" and self.current_phase == Phase.PICKING:
            print("Pour task failed（夹爪闭合不足）！保存失败数据...")
        else:
            print("保存失败数据...")
        self.data_collector.write_cached_data(state['joint_positions'][:-1])
        self._last_success = False
        self.current_phase = Phase.FINISHED
        return None, True, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.pick_controller.is_done():
            action = None
            action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 15])).as_quat(),
                )
            
        else:
            state['language_instruction'] = self.get_language_instruction()

            action = self.inference_engine.step_inference(state)
        success = self._check_phase_success()
        if success and self.current_phase == Phase.PICKING:
            print("Pick task success! Switching to pour...")
            self.current_phase = Phase.POURING
        elif success and self.current_phase == Phase.POURING:
            print("Pour task success!")
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return None, True, True
               
        return action, False, False

    def _maybe_mark_pick_exception(self, state):
        """记录夹取阶段相关异常用于截图标注/命名。"""
        info = None
        if self.failure_mode == "fail_angle_pick" and hasattr(self.pick_controller, "get_angle_deviation_info"):
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

    def _maybe_mark_pour_exception(self, state):
        """记录倾倒阶段相关异常用于截图标注/命名。"""
        info = None
        if self.failure_mode == "drop" and hasattr(self.pour_controller, "get_drop_info"):
            info = self.pour_controller.get_drop_info()
        elif self.failure_mode == "fail_position" and hasattr(self.pour_controller, "get_fail_position_info"):
            info = self.pour_controller.get_fail_position_info()
        elif self.failure_mode == "incomplete_angle" and hasattr(self.pour_controller, "get_incomplete_angle_info"):
            info = self.pour_controller.get_incomplete_angle_info()

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

    def get_language_instruction(self) -> Optional[str]:
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').lower()
        self.language_instruction = f"Pick up the {object_name} from the table and pour it into the target".replace("  ", " ")
        return self.language_instruction