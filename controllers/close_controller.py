import re
from typing import Optional
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
import numpy as np
from scipy.spatial.transform import Rotation as R
from controllers.atomic_actions.close_controller import CloseController
from controllers.atomic_actions.close_controller_fail_position import CloseControllerFailPosition
from .base_controller import BaseController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from .inference_engines.inference_engine_factory import InferenceEngineFactory

class CloseTaskController(BaseController):
    """Controller for managing the task of closing a door in collect or infer mode.

    Args:
        cfg: Configuration object containing mode and other parameters.
        robot: Robot articulation instance.
    """

    def __init__(self, cfg, robot):
        self.operate_type = cfg.task.get("operate_type", "door")
        print(self.operate_type)
        # Failure mode: "incomplete_close", "fail_position" or None
        # 必须在 super().__init__ 之前初始化，因为 _init_collect_mode 会用到它
        self.failure_mode = cfg.task.get("failure_mode", None)
        super().__init__(cfg, robot)
        self.initial_handle_position = None
        # For incomplete_close mode: early stop and wait
        self._early_stop_triggered = False
        self._wait_steps = 0
        self._wait_steps_required = cfg.task.get("wait_steps_after_stop", 100)  # Default 100 steps (~1.2 seconds at 60fps)
        self._frame_exception = None
            
    def _init_collect_mode(self, cfg, robot):
        """Initializes the controller for data collection mode.

        Args:
            cfg: Configuration object for collect mode.
            robot: Robot articulation instance.
        """
        super()._init_collect_mode(cfg, robot)
        
        if self.failure_mode == "fail_position":
            position_offset = cfg.task.get("position_offset", 0.05)
            self.close_controller = CloseControllerFailPosition(
                name="close_controller",
                cspace_controller=RMPFlowController(
                    name="target_follower_controller",
                    robot_articulation=robot
                ),
                gripper=robot.gripper,
                furniture_type=self.operate_type,
                door_open_direction="clockwise",
                position_offset=position_offset
            )
        else:
            self.close_controller = CloseController(
                name="close_controller",
                cspace_controller=RMPFlowController(
                    name="target_follower_controller",
                    robot_articulation=robot
                ),
                gripper=robot.gripper,
                furniture_type=self.operate_type,
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
        if self.mode == "collect":
            self.close_controller.reset()
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
        """Executes a step in collect mode using the close controller.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        # Check if we should trigger early stop (for incomplete_close mode)
        if (self.failure_mode == "incomplete_close" and 
            not self._early_stop_triggered and 
            self._check_early_stop_threshold(state)):
            operate_type = self.operate_type
            object_name = "Drawer" if operate_type == "drawer" else "Door"
            print(f"Early stop triggered: incomplete close {object_name.lower()} (threshold)")
            self._early_stop_triggered = True
            # 标记不完全关门/抽屉异常，用于命名/截图
            self._frame_exception = {
                "type": "incomplete_close",
                "message": f"{object_name} not fully closed (early stop)",
                "suffix": "erro",
                "color": (0, 0, 255),
                "frame_idx": state.get("frame_idx"),
                "episode": self._episode_num,
            }
            # Force close_controller to be done by resetting it
            self.close_controller.reset()
            # Set it to done state by advancing to final event
            if hasattr(self.close_controller, '_event'):
                self.close_controller._event = len(self.close_controller._events_dt)
        
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
                print(f"Task completed with incomplete close (waited {self._wait_steps} steps)!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.reset_needed = True
                return None, True, True
            else:
                # Still waiting, return no action
                return None, False, False
        
        # Normal execution: continue with close controller
        if not self.close_controller.is_done():
            if self.operate_type == "door":
                action = self.close_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    revolute_joint_position=state['revolute_joint_position'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([350, 90, 25])).as_quat(),
                )
            else:
                action = self.close_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=euler_angles_to_quats([90, 90, 0], degrees=True, extrinsic=False),
                    push_distance=0.15
                )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            # 检查并标记推门位置错误异常
            self._maybe_mark_close_exception(state)
            
            if self._check_success(state):
                self.check_success_counter += 1
            else:
                self.check_success_counter = 0
                
            return action, False, False

        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
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
            state['language_instruction'] = "Close the drawer of the object"
        
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
        
    def _check_early_stop_threshold(self, state):
        """Checks if early stop threshold is reached (for incomplete_close mode).

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if early stop threshold is reached, False otherwise.
        """
        current_pos = state['object_position']
        gripper_position = state['gripper_position']
        
        if self.operate_type == "drawer":
            # For drawer, check if moved enough to trigger early stop
            return (
                np.linalg.norm(np.array(current_pos) - self.initial_handle_position) > 0.08 and
                np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.04
            )
        else:
            # For door, check if moved enough to trigger early stop
            # Need to ensure door has been pushed (moved at least 0.02m) but stop before fully closed (success threshold is 0.08m)
            # Check that close_controller has started pushing (event >= 1, meaning it has reached the handle and started pushing)
            door_moved_distance = np.array(current_pos)[0] - self.initial_handle_position[0]
            
            # Check if close_controller has started pushing (event 1 means it's in pushing phase)
            controller_started_pushing = False
            if hasattr(self.close_controller, '_event'):
                # event 0: approaching handle, event 1: pushing door, event 2: retracting
                controller_started_pushing = self.close_controller._event >= 1
            
            # Trigger early stop only if:
            # 1. Controller has started pushing (event >= 1)
            # 2. Door has moved at least 0.02m (some movement occurred)
            # 3. Door hasn't reached success threshold (0.08m)
            return (
                controller_started_pushing and 
                door_moved_distance > 0.02 and 
                door_moved_distance < 0.08
            )
    
    def _check_success(self, state):
        """Checks if the task has been successfully completed.

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        current_pos = state['object_position']
        gripper_position = state['gripper_position']
        
        # Normal success check
        if self.operate_type == "drawer":
            return (
                np.linalg.norm(np.array(current_pos) - self.initial_handle_position) > 0.13 and
                np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.04
            )
        else:
            return (
                np.array(current_pos)[0] - self.initial_handle_position[0] > 0.08 and
                np.linalg.norm(np.array(gripper_position) - np.array(current_pos)) > 0.08
            )

    def _maybe_mark_close_exception(self, state):
        """检查并标记关门异常（如推门位置错误）。"""
        if self.failure_mode == "fail_position":
            info = self.close_controller.get_fail_position_info()
            if info is not None:
                self._frame_exception = {
                    **info,
                    "frame_idx": state.get("frame_idx"),
                    "episode": self._episode_num,
                }

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
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').lower()
        if self.failure_mode == "incomplete_close":
            # For incomplete close, we still use the same instruction to collect failure data
            self._language_instruction = f"Close the {self.operate_type} of the {object_name}"
        else:
            self._language_instruction = f"Close the {self.operate_type} of the {object_name}"
        return self._language_instruction
