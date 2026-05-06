from typing import Optional
import numpy as np

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
#from .atomic_actions.stir_controller import StirController
from .atomic_actions.stir_controller import StirController
#from .atomic_actions.stir_controller_slip import StirController
class StirTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        # Failure mode: "fail_position", "fail_angle", "fail_position_pick", "fail_angle_pick", "incomplete_close_pick", "drop" or None
        self.failure_mode = OmegaConf.select(cfg, "task.failure_mode", default=None)
        self._frame_exception = None
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        # 根据 failure_mode 选择不同的 pick 控制器
        failure_mode = OmegaConf.select(cfg, "task.failure_mode", default=None)
        if failure_mode == "fail_position_pick":
            from controllers.atomic_actions.pick_controller_fail_position import PickControllerFailPosition
            position_offset = OmegaConf.select(cfg, "task.position_offset", default=0.05)  # 默认偏移0.05米
            self.pick_controller = PickControllerFailPosition(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                position_threshold=0.005,
                events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02],
                position_offset=position_offset
            )
        elif failure_mode == "fail_angle_pick":
            from controllers.atomic_actions.pick_controller_fail_angle import PickControllerFailAngle
            angle_offset_deg = OmegaConf.select(cfg, "task.angle_offset_deg", default=45.0)
            deviation_report_threshold_deg = OmegaConf.select(cfg, "task.angle_report_threshold_deg", default=20.0)
            angle_offset_range_deg = OmegaConf.select(cfg, "task.angle_offset_range_deg", default=None)
            angle_offset_choices_deg = OmegaConf.select(cfg, "task.angle_offset_choices_deg", default=None)
            self.pick_controller = PickControllerFailAngle(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                position_threshold=0.005,
                events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02],
                angle_offset_deg=angle_offset_deg,
                angle_offset_range_deg=angle_offset_range_deg,
                angle_offset_choices_deg=angle_offset_choices_deg,
                deviation_report_threshold_deg=deviation_report_threshold_deg
            )
        elif failure_mode == "incomplete_close_pick":
            from controllers.atomic_actions.pick_controller_incomplete_close import PickControllerIncompleteClose
            incomplete_close_scale = OmegaConf.select(cfg, "task.incomplete_close_scale", default=1.6)
            extra_gap = OmegaConf.select(cfg, "task.extra_gap", default=0.0)
            gap_report_threshold = OmegaConf.select(cfg, "task.gap_report_threshold", default=0.002)
            self.pick_controller = PickControllerIncompleteClose(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                position_threshold=0.005,
                events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02],
                incomplete_close_scale=incomplete_close_scale,
                extra_gap=extra_gap,
                gap_report_threshold=gap_report_threshold
            )
        else:
            # Pick 阶段使用正常的 PickController（夹取阶段正常完成）
            self.pick_controller = PickController(
                name="pick_controller",
                cspace_controller=self.rmp_controller,
                position_threshold=0.005,
                events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02]
            )
        
        # 根据 failure_mode 选择不同的 stir 控制器
        failure_mode = OmegaConf.select(cfg, "task.failure_mode", default=None)
        if failure_mode == "fail_position":
            from controllers.atomic_actions.stir_controller_fail_position import StirControllerFailPosition
            position_offset = OmegaConf.select(cfg, "task.position_offset", default=0.05)  # 默认偏移0.05米
            self.stir_controller = StirControllerFailPosition(
                name="stir_controller",
                cspace_controller=self.rmp_controller,
                position_offset=position_offset
            )
        elif failure_mode == "fail_angle":
            from controllers.atomic_actions.stir_controller_fail_angle import StirControllerFailAngle
            angle_offset_deg = OmegaConf.select(cfg, "task.angle_offset_deg", default=15.0)  # 默认角度偏移15度
            self.stir_controller = StirControllerFailAngle(
                name="stir_controller",
                cspace_controller=self.rmp_controller,
                angle_offset_deg=angle_offset_deg
            )
        elif failure_mode == "drop":
            from controllers.atomic_actions.stir_controller_drop import StirControllerDrop
            self.stir_controller = StirControllerDrop(
                name="stir_controller",
                cspace_controller=self.rmp_controller,
            )
        elif failure_mode == "no_pick":
            from controllers.atomic_actions.stir_controller_no_pick import StirControllerNoPick
            self.stir_controller = StirControllerNoPick(
                name="stir_controller",
                cspace_controller=self.rmp_controller,
            )
        elif failure_mode == "no_stir":
            from controllers.atomic_actions.stir_controller_no_stir import StirControllerNoStir
            self.stir_controller = StirControllerNoStir(
                name="stir_controller",
                cspace_controller=self.rmp_controller,
            )
        else:
            self.stir_controller = StirController(
                name="stir_controller",
                cspace_controller=self.rmp_controller,
            )

        self.gripper_control.release_object()
        self._last_pick_joint_data = None

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt = [0.004, 0.002, 0.005, 1, 0.05, 0.004, 1]
        )
        self.use_stir_model = False
        self.frame_count = 0
        
    def reset(self):
        super().reset()
        self.gripper_control.release_object()
        self.pick_controller.reset()
        if self.mode == "collect":
            self.stir_controller.reset()
        else:
            self.inference_engine.reset()
        self.initial_position = None
        self.use_stir_model = False
        self.frame_count = 0
        self._frame_exception = None
    
    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
        
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        # 对于 no_pick 模式，跳过 pick 阶段，直接进入 stir 阶段
        if self.failure_mode == "no_pick":
            # 跳过 pick 阶段，直接执行 stir
            target_position = self.object_utils.get_object_xform_position(
                object_path=state['target_beaker']
            )
            if target_position is None:
                target_position = state['target_position']
            
            action = self.stir_controller.forward(
                center_position=target_position,
                current_joint_positions=state['joint_positions'],
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, -10])).as_quat(),
            )
            # no_pick 模式下不需要更新夹取物体位置（因为没有夹取物体）
            
            # 检查并标记没有夹取物体就开始搅拌的异常
            self._maybe_mark_stir_exception(state)
            
            # 缓存步骤数据
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            if self.stir_controller.is_done():
                return action, True, False
            else:
                return action, False, False
        
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                after_offset_z= 0.15,
                gripper_distances=0.005
            )
            
            self.gripper_control.update_grasped_object_position()
            
            # 检查并标记夹取位置错误异常
            self._maybe_mark_pick_exception(state)
            
            # 缓存步骤数据（pick阶段也需要）
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )

            return action, False, False
            
        elif not self.stir_controller.is_done():
            target_position = self.object_utils.get_object_xform_position(
                object_path=state['target_beaker']
            )
            if target_position is None:
                target_position = state['target_position']
            
            # 对于 drop 模式，需要传递 gripper_control 参数
            if self.failure_mode == "drop":
                action = self.stir_controller.forward(
                    center_position=target_position,
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, -10])).as_quat(),
                )
                # 只有在玻璃棒还没有掉落时才更新位置
                if self.gripper_control.grasped_object_path is not None:
                    self.gripper_control.update_grasped_object_position()
            else:
                action = self.stir_controller.forward(
                    center_position=target_position,
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, -10])).as_quat(),
                )
                self.gripper_control.update_grasped_object_position()
            
            # 检查并标记搅拌异常（如没有执行搅拌动作）
            self._maybe_mark_stir_exception(state)
            
            # 缓存步骤数据（stir阶段）
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
                
            return action, False, False
            
        else:
            self.reset_needed = True
            final_object_position = state['glass_rod_position']
            target_position = state['target_position']
            self.gripper_control.release_object()
            
            # 检查任务是否成功
            success = (final_object_position is not None and 
                      final_object_position[2] > 0.85 and
                      np.linalg.norm(final_object_position[0:2] - target_position[0:2]) < 0.04)
            
            # 在写入数据前检查是否已达到最大 episode 数
            if self.data_collector.episode_count >= self.data_collector.max_episodes:
                print(f"Reached max_episodes ({self.data_collector.max_episodes}), stopping data collection.")
                self._last_success = False
                return None, True, False
            
            # 对于 drop 模式，保存失败数据（物体在移动过程中掉落）
            if self.failure_mode == "drop":
                if not success:
                    print("Stir task failed（玻璃棒在移动过程中掉落）！保存失败数据...")
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = False
                    return None, True, False
                else:
                    print("Stir task success!")
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = True
                    return None, True, True
            else:
                # 正常模式的处理
                if success:
                    print("Stir task success!")
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = True
                    return None, True, True
                else:
                    # 保存失败数据，而不是清除缓存
                    print("Stir task failed! 保存失败数据...")
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = False
                    return None, True, False

    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Processes observations and generates control actions using learned policy.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                after_offset_z=0.15,
            )
            
            final_object_position = state['glass_rod_position']
            if final_object_position is not None and final_object_position[2] > 0.82:
                self.use_stir_model = True

            self.gripper_control.update_grasped_object_position()

            return action, False, False
        
        state['language_instruction'] = self.get_language_instruction()
        
        if self.use_stir_model:
            action = self.inference_engine.step_inference(state)
            self.gripper_control.update_grasped_object_position()

            return action, False, self._check_success()
        
        return ArticulationAction(), False, False
    
    def _check_success(self):
        object_pos = self.state['glass_rod_position']
        target_position = self.state['target_position']
        if object_pos[2] > 0.85 and np.linalg.norm(object_pos[0:2] - target_position[0:2]) < 0.04:
            self.check_success_counter += 1
            if self.check_success_counter > 240:
                self._last_success = True
                return True
        return False

    def _maybe_mark_pick_exception(self, state):
        """检查并标记夹取异常（如夹取位置错误、角度错误、夹爪未完全闭合）。"""
        if self.failure_mode == "fail_position_pick":
            if hasattr(self.pick_controller, 'get_position_error_info'):
                info = self.pick_controller.get_position_error_info()
                if info is not None:
                    self._frame_exception = {
                        **info,
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                    }
        elif self.failure_mode == "fail_angle_pick":
            if hasattr(self.pick_controller, 'get_angle_deviation_info'):
                info = self.pick_controller.get_angle_deviation_info()
                if info is not None:
                    self._frame_exception = {
                        **info,
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                    }
        elif self.failure_mode == "incomplete_close_pick":
            if hasattr(self.pick_controller, 'get_incomplete_close_info'):
                info = self.pick_controller.get_incomplete_close_info()
                if info is not None:
                    # 确保使用 erro 后缀
                    if info.get("suffix") != "erro":
                        info["suffix"] = "erro"
                    self._frame_exception = {
                        **info,
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                    }

    def _maybe_mark_stir_exception(self, state):
        """检查并标记搅拌异常（如没有夹取物体就开始搅拌、没有执行搅拌动作、插入角度错误）。"""
        if self.failure_mode == "no_pick":
            if hasattr(self.stir_controller, 'get_no_pick_info'):
                info = self.stir_controller.get_no_pick_info()
                if info is not None:
                    self._frame_exception = {
                        **info,
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                    }
        elif self.failure_mode == "no_stir":
            if hasattr(self.stir_controller, 'get_no_stir_info'):
                info = self.stir_controller.get_no_stir_info()
                if info is not None:
                    self._frame_exception = {
                        **info,
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                    }
        elif self.failure_mode == "fail_angle":
            if hasattr(self.stir_controller, 'get_fail_angle_info'):
                info = self.stir_controller.get_fail_angle_info()
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
        # Default instruction for shake tasks
        self._language_instruction = "Use the glass rod to stir the liquid."
        return self._language_instruction
