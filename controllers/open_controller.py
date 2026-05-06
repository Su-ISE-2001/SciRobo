import re
from typing import Optional
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
import numpy as np

from controllers.atomic_actions.open_controller import OpenController
from .base_controller import BaseController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from .inference_engines.inference_engine_factory import InferenceEngineFactory

class OpenTaskController(BaseController):
    """Controller for managing the task of opening a door in collect or infer mode.

    Args:
        cfg: Configuration object containing mode and other parameters.
        robot: Robot articulation instance.
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_handle_position = None
        # Failure mode: "incomplete_open" or "fail_grasp" or None
        self.failure_mode = cfg.task.get("failure_mode", None)
        # For incomplete_open mode: early stop and wait
        self._early_stop_triggered = False
        self._wait_steps = 0
        self._wait_steps_required = cfg.task.get("wait_steps_after_stop", 100)  # Default 100 steps (~1.2 seconds at 60fps)
        self._frame_exception = None
        # 可选：按固定步数触发早停（优先于角度阈值），None 表示不用步数控制
        self._early_stop_steps = cfg.task.get("early_stop_steps", None)
        self._open_step_counter = 0
            
    def _init_collect_mode(self, cfg, robot):
        """Initializes the controller for data collection mode.

        Args:
            cfg: Configuration object for collect mode.
            robot: Robot articulation instance.
        """
        super()._init_collect_mode(cfg, robot)
        
        # Use different controllers based on failure mode
        # Read from cfg directly since self.failure_mode may not be set yet
        failure_mode = cfg.task.get("failure_mode", None)
        if failure_mode == "fail_grasp":
            from controllers.atomic_actions.open_controller_fail_grasp import OpenControllerFailGrasp
            self.open_controller = OpenControllerFailGrasp(
                name="open_controller",
                cspace_controller=RMPFlowController(
                    name="target_follower_controller",
                    robot_articulation=robot
                ),
                gripper=robot.gripper,
                events_dt=[0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008],
                furniture_type=self.cfg.task.get("operate_type", "door"),
                door_open_direction="clockwise"
            )
        else:
            self.open_controller = OpenController(
                name="open_controller",
                cspace_controller=RMPFlowController(
                    name="target_follower_controller",
                    robot_articulation=robot
                ),
                gripper=robot.gripper,
                events_dt=[0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008],
                furniture_type=self.cfg.task.get("operate_type", "door"),
                door_open_direction="clockwise"
            )

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

    def reset(self):
        """Resets the controller to its initial state."""
        super().reset()
        self.initial_handle_position = None
        self._early_stop_triggered = False
        self._wait_steps = 0
        self._frame_exception = None
        self._open_step_counter = 0
        if self.mode == "collect":
            self.open_controller.reset()
        else:
            self.inference_engine.reset()

    def step(self, state):
        """Executes one step of the task based on the current state.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        self.state = state
        if self.initial_handle_position is None:
            self.initial_handle_position = state['object_position']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Executes a step in collect mode using the open controller.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        # Check if we should trigger early stop (for incomplete_open mode)
        if (self.failure_mode == "incomplete_open" and 
            not self._early_stop_triggered and 
            (
                (self._early_stop_steps is not None and self._open_step_counter >= self._early_stop_steps)
                or self._check_early_stop_threshold(state)
            )):
            operate_type = self.cfg.task.get("operate_type", "door")
            object_name = "Drawer" if operate_type == "drawer" else "Door"
            print(f"Early stop triggered: incomplete open {object_name.lower()} (step or threshold)")
            self._early_stop_triggered = True
            # 标记不完全开门/抽屉异常，用于命名/截图
            self._frame_exception = {
                "type": "incomplete_open",
                "message": f"{object_name} not fully opened (early stop)",
                "suffix": "erro",
                "color": (0, 0, 255),
                "frame_idx": state.get("frame_idx"),
                "episode": self._episode_num,
            }
            # Force open_controller to be done by resetting it
            self.open_controller.reset()
            # Set it to done state by advancing to final event
            if hasattr(self.open_controller, '_event'):
                self.open_controller._event = len(self.open_controller._events_dt)
        
        # If early stop is triggered, enter waiting phase
        if self._early_stop_triggered:
            self._wait_steps += 1
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            # Wait for required steps, then mark as success
            if self._wait_steps >= self._wait_steps_required:
                print(f"Task completed with incomplete open (waited {self._wait_steps} steps)!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.reset_needed = True
                return None, True, True
            else:
                # Still waiting, return no action
                return None, False, False
        
        # Normal execution: continue with open controller
        if not self.open_controller.is_done():
            if self.cfg.task.get("operate_type") == "door":
                action = self.open_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    revolute_joint_position=state['revolute_joint_position'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=euler_angles_to_quats([0, 110, 0], degrees=True, extrinsic=False),
                )
            else:
                action = self.open_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=euler_angles_to_quats([90, 90, 0], degrees=True, extrinsic=False),
                )
            # 递增步计数（仅在未触发早停且处于 incomplete_open 模式）
            if self.failure_mode == "incomplete_open" and not self._early_stop_triggered:
                self._open_step_counter += 1
            # 标记异常（fail_grasp 模式）
            self._maybe_mark_open_exception(state)
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            if self._check_success(state):
                self.check_success_counter += 1
            else:
                self.check_success_counter = 0
                
            return action, False, False

        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        
        # For fail_grasp mode, we want to save the failure data
        if self.failure_mode == "fail_grasp":
            if not success:
                print("Task failed (grasp failure)! Saving failure data...")
                # Save the failure data for training on failure cases
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = False
            else:
                print("Task success!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
        else:
            if success:
                print("Task success!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
            else:
                print("Task failed!")
                self.data_collector.clear_cache()
                self._last_success = False
            
        self.reset_needed = True
        return None, True, success

    def _step_infer(self, state):
        """Executes a step in infer mode using the trained policy.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Open the door of the object"
        
        action = self.inference_engine.step_inference(state)
        
        if self._check_success(state):
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            print("Task success!")
            self._last_success = True
            self.reset_needed = True
            return None, True, True
            
        return action, False, False

    def consume_frame_exception(self):
        """供外部读取并清空当前帧异常标记。"""
        info = self._frame_exception
        self._frame_exception = None
        return info
        
    def _check_early_stop_threshold(self, state):
        """Checks if early stop threshold is reached (for incomplete_open mode).

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if early stop threshold is reached, False otherwise.
        """
        current_pos = state['object_position']
        gripper_position = state['gripper_position']
        operate_type = self.cfg.task.get("operate_type", "door")
        
        if operate_type == "drawer":
            # For drawer, check if moved enough to trigger early stop
            return (
                np.linalg.norm(np.array(current_pos) - self.initial_handle_position) > 0.08 and
                np.linalg.norm(np.array(current_pos) - self.initial_handle_position) < 0.13 and
                np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.04
            )
        else:
            # 对"门"使用转角阈值判定：当把手转角进入指定区间时触发早停
            # 默认区间：0.2~0.35 rad，可通过 cfg.task.early_stop_angle_range_rad 覆盖
            angle_range = self.cfg.task.get("early_stop_angle_range_rad", [0.2, 0.35])
            try:
                lo, hi = float(angle_range[0]), float(angle_range[1])
            except Exception:
                lo, hi = 0.2, 0.35
            joint_angle_raw = state.get("revolute_joint_position", state.get("revolute_joint_positions", 0.0))
            # 确保 joint_angle 是标量（处理数组情况）
            if isinstance(joint_angle_raw, (np.ndarray, list, tuple)):
                joint_angle = abs(float(np.array(joint_angle_raw).flatten()[0]))
            else:
                joint_angle = abs(float(joint_angle_raw))
            if self.cfg.task.get("early_stop_debug", False):
                print(f"[incomplete_open] joint_angle={joint_angle:.4f} rad, range=({lo:.3f}, {hi:.3f})")
            return (joint_angle >= lo) and (joint_angle <= hi)
    
    def _check_success(self, state):
        """Checks if the task has been successfully completed.

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        current_pos = state['object_position']
        gripper_position = state['gripper_position']
        operate_type = self.cfg.task.get("operate_type", "door")
        
        if operate_type == "drawer":
            return (
                np.linalg.norm(np.array(current_pos) - self.initial_handle_position) > 0.13 and
                np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.04
            )
        else:
            # For door, check x-direction movement (consistent with close_controller)
            # Check absolute value since door can open in either direction
            x_displacement = abs(np.array(current_pos)[0] - self.initial_handle_position[0])
            return (
                x_displacement > 0.08 and
                np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.04
            )

    def _maybe_mark_open_exception(self, state):
        """记录开门相关异常用于截图标注/命名。"""
        info = None
        if self.failure_mode == "fail_grasp" and hasattr(self.open_controller, "get_fail_grasp_info"):
            info = self.open_controller.get_fail_grasp_info()
        elif self.failure_mode == "incomplete_open" and self._early_stop_triggered and self._frame_exception is None:
            # incomplete_open 的异常在早停触发时已设置，这里不再重复设置
            pass

        if info is None:
            return
        info["frame_idx"] = state.get("frame_idx")
        info["episode"] = self._episode_num
        self._frame_exception = info

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').lower()
        if self.failure_mode == "incomplete_open":
            # For incomplete open, we still use the same instruction to collect failure data
            self._language_instruction = f"Open the door of the {object_name}"
        elif self.failure_mode == "fail_grasp":
            # For fail_grasp, we still use the same instruction to collect failure data
            self._language_instruction = f"Open the door of the {object_name}"
        else:
            self._language_instruction = f"Open the door of the {object_name}"
        return self._language_instruction
