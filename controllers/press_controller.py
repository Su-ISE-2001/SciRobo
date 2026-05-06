from typing import Optional
import numpy as np

from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.press_controller import PressController

class PressTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        # Failure mode: "fail_position", "incomplete" or None
        self.failure_mode = cfg.task.get("failure_mode", None)
        self._frame_exception = None
        
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        # Use different controllers based on failure mode
        failure_mode = cfg.task.get("failure_mode", None)
        if failure_mode == "fail_position":
            from controllers.atomic_actions.press_controller_fail_position import PressControllerFailPosition
            position_offset = cfg.task.get("position_offset", 0.03)  # 默认偏移0.03米
            self.press_controller = PressControllerFailPosition(
                name="press_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                events_dt=[0.005, 0.1, 0.005],
                position_offset=position_offset
            )
        elif failure_mode == "incomplete":
            from controllers.atomic_actions.press_controller_incomplete import PressControllerIncomplete
            press_distance_reduction = cfg.task.get("press_distance_reduction", 0.15)  # 默认只按15%的距离
            self.press_controller = PressControllerIncomplete(
                name="press_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                events_dt=[0.005, 0.1, 0.005],
                press_distance_reduction=press_distance_reduction
            )
        else:
            self.press_controller = PressController(
                name="press_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                events_dt = [0.005, 0.1, 0.005]  # Default phase durations
            )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.press_controller.reset()
        else:
            self.inference_engine.reset()
        self._frame_exception = None
    
    def step(self, state):
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        final_object_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )
        return final_object_position is not None and final_object_position[0] > 0.405

    def _step_collect(self, state):
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.press_controller.is_done():
            action = self.press_controller.forward(
                target_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            self._maybe_mark_press_exception(state)
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False
        
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        
        # For different failure modes, we want to save the failure data
        if self.failure_mode == "fail_position":
            if not self._last_success:
                print("任务失败（按按钮位置错误）！保存失败数据...")
                # 保存失败数据用于训练失败案例
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = False
            else:
                print("任务成功！")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
        elif self.failure_mode == "incomplete":
            # For incomplete mode, we want to save the failure data (按钮未被完全按下)
            if not self._last_success:
                print("任务失败（按钮未被完全按下）！保存失败数据...")
                # 保存失败数据用于训练失败案例
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = False
            else:
                print("任务成功！")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
        else:
            if self._last_success:
                print("任务成功！")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
            else:
                print("任务失败！")
                self.data_collector.clear_cache()
                self._last_success = False
            
        self.reset_needed = True
        return None, True, self._last_success
        
    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction

        action = self.inference_engine.step_inference(state)
        
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        self._maybe_mark_press_exception(state)
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        if self._language_instruction is None:
            return "Press the different color button"
        return self._language_instruction

    def _maybe_mark_press_exception(self, state):
        """记录按压阶段异常用于截图标注/命名。"""
        info = None
        if self.failure_mode == "incomplete" and hasattr(self.press_controller, "get_incomplete_press_info"):
            info = self.press_controller.get_incomplete_press_info()
        elif self.failure_mode == "fail_position" and hasattr(self.press_controller, "get_fail_position_info"):
            info = self.press_controller.get_fail_position_info()

        if info is None:
            return
        info["frame_idx"] = state.get("frame_idx")
        info["episode"] = self._episode_num
        self._frame_exception = info

    def consume_frame_exception(self):
        """供主循环读取并清空当前帧异常标记。"""
        info = self._frame_exception
        self._frame_exception = None
        return info
