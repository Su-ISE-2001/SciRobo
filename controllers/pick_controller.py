import re
from typing import Optional
import numpy as np
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import random

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from .inference_engines.inference_engine_factory import InferenceEngineFactory
class PickTaskController(BaseController):
    """
    Controller for pick-and-place tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous picking

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        REQUIRED_SUCCESS_STEPS (int): Number of consecutive steps needed for success
        success_counter (int): Counter for tracking successful steps
    """
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        # Failure mode: "fail_position", "drop" or None
        self.failure_mode = cfg.task.get("failure_mode", None)
        self._frame_exception = None
        
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up pick controller, gripper control, and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)
        
        # Use different controllers based on failure mode
        failure_mode = cfg.task.get("failure_mode", None)
        if failure_mode == "fail_position":
            from controllers.atomic_actions.pick_controller_fail_position import PickControllerFailPosition
            position_offset = cfg.task.get("position_offset", 0.03)  # 默认偏移0.03米
            self.pick_controller = PickControllerFailPosition(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008],
                position_offset=position_offset
            )
        elif failure_mode == "fail_angle":
            from controllers.atomic_actions.pick_controller_fail_angle import PickControllerFailAngle
            angle_offset_deg = cfg.task.get("angle_offset_deg", 30.0)
            deviation_report_threshold_deg = cfg.task.get("angle_report_threshold_deg", 20.0)
            angle_offset_range_deg = cfg.task.get("angle_offset_range_deg", None)
            angle_offset_choices_deg = cfg.task.get("angle_offset_choices_deg", None)
            self.pick_controller = PickControllerFailAngle(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008],
                angle_offset_deg=angle_offset_deg,
                angle_offset_range_deg=angle_offset_range_deg,
                angle_offset_choices_deg=angle_offset_choices_deg,
                deviation_report_threshold_deg=deviation_report_threshold_deg,
            )
        elif failure_mode == "incomplete_close":
            from controllers.atomic_actions.pick_controller_incomplete_close import PickControllerIncompleteClose
            incomplete_close_scale = cfg.task.get("incomplete_close_scale", 1.5)
            extra_gap = cfg.task.get("extra_gap", 0.0)
            gap_report_threshold = cfg.task.get("gap_report_threshold", 0.002)
            self.pick_controller = PickControllerIncompleteClose(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008],
                incomplete_close_scale=incomplete_close_scale,
                extra_gap=extra_gap,
                gap_report_threshold=gap_report_threshold,
            )
        elif failure_mode == "drop":
            from controllers.atomic_actions.pick_controller_drop import PickControllerDrop
            self.pick_controller = PickControllerDrop(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.05, 0.004, 0.008]
            )
        else:
            self.pick_controller = PickController(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
            )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_controller.reset()
        else:
            self.inference_engine.reset()
        self.initial_position = None
        self._frame_exception = None
    
    def step(self, state):
        # 检查必要的状态信息是否存在
        if state is None or state.get('object_position') is None:
            # 如果状态或对象位置不可用，返回空动作并继续等待
            return None, False, False
        
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self, state=None):
        """检查任务是否成功完成。
        
        Args:
            state: 环境状态字典，如果为None则使用self.state
        
        Returns:
            bool: 如果物体被成功抬起则返回True
        """
        if state is None:
            state = self.state
        if state is None or self.initial_position is None:
            return False
        return state['object_position'][2] > self.initial_position[2] + 0.1

    def _init_infer_mode(self, cfg, robot):
        """
        Initializes components for inference mode.
        Creates inference engine and trajectory controller.

        Args:
            cfg: Configuration object containing model paths and settings
            robot: Robot instance to control
        """
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )
        
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        # 检查必要的状态信息是否存在
        if state.get('object_position') is None:
            # 如果对象位置不可用，返回空动作并继续等待
            return None, False, False
        
        if self._check_success(state):
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.pick_controller.is_done():
            # 对于 drop 模式，设置机器人位置以提高接近方向计算的准确性
            if self.failure_mode == "drop" and hasattr(self.pick_controller, 'set_robot_position'):
                robot_position = np.array(self.cfg.robot.position) if hasattr(self.cfg, 'robot') and hasattr(self.cfg.robot, 'position') else None
                if robot_position is not None:
                    self.pick_controller.set_robot_position(robot_position)
            
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_offset_x=0.05,
                after_offset_z=0.25
            )

            self._maybe_mark_pick_exception(state)
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False
        
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        
        # 在写入数据前检查是否已达到最大 episode 数
        if self.data_collector.episode_count >= self.data_collector.max_episodes:
            print(f"Reached max_episodes ({self.data_collector.max_episodes}), stopping data collection.")
            self._last_success = False
            self.reset_needed = True
            return None, True, False
        
        # For fail_position mode, we want to save the failure data
        if self.failure_mode == "fail_position":
            if not self._last_success:
                print("任务失败（夹取位置错误）！保存失败数据...")
                # 保存失败数据用于训练失败案例
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = False
            else:
                print("任务成功！")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
        elif self.failure_mode == "drop":
            # For drop mode, we want to save the failure data (物体在抬起过程中掉落)
            if not self._last_success:
                print("任务失败（夹取时掉落）！保存失败数据...")
                # 保存失败数据用于训练失败案例
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = False
            else:
                print("任务成功！")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
        else:
            # 默认模式：无论成功或失败都写出数据，不再丢弃失败数据
            if self._last_success:
                print("任务成功！保存数据。")
                self._last_success = True
            else:
                print("任务失败！依然保存数据。")
                self._last_success = False
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            
        self.reset_needed = True
        return None, True, self._last_success
        
    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Uses inference engine to process observations and generate actions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
            
        action = self.inference_engine.step_inference(state)
        
        if self._check_success(state):
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        return action, False, False

    def _maybe_mark_pick_exception(self, state):
        """记录夹取相关的异常用于截图标注。"""
        info = None
        if self.failure_mode == "fail_angle" and hasattr(self.pick_controller, "get_angle_deviation_info"):
            info = self.pick_controller.get_angle_deviation_info()
        elif self.failure_mode == "incomplete_close" and hasattr(self.pick_controller, "get_incomplete_close_info"):
            info = self.pick_controller.get_incomplete_close_info()
        elif self.failure_mode == "drop" and hasattr(self.pick_controller, "get_drop_info"):
            info = self.pick_controller.get_drop_info()
        elif self.failure_mode == "fail_position" and hasattr(self.pick_controller, "get_position_error_info"):
            info = self.pick_controller.get_position_error_info()

        if info is None:
            return
        info["frame_idx"] = state.get("frame_idx")
        info["episode"] = self._episode_num
        self._frame_exception = info

    def consume_frame_exception(self):
        """供外部（如主循环）读取并清空当前帧异常标记。"""
        info = self._frame_exception
        self._frame_exception = None
        return info

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ', ' ').lower()
        self._language_instruction = f"Pick up the {object_name} from the table"
        return self._language_instruction
