import re
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum

from .atomic_actions.pick_controller_aim import PickController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    FINISHED = "finished"

class PlaceTaskAimController(BaseController):
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
        # Exception info consumed by main loop for screenshot naming
        self._frame_exception = None

    def consume_frame_exception(self):
        """Return and clear per-frame exception info for screenshot naming/annotation."""
        info = self._frame_exception
        self._frame_exception = None
        return info

    def _set_frame_exception_once(self, message: str = "erro", persist_suffix: str = "erro", suffix: str = "", color=(0, 0, 255)):
        """Set exception info once (avoid spamming)."""
        if self._frame_exception is not None:
            return
        self._frame_exception = {
            "message": message,
            "persist_suffix": persist_suffix,
            "suffix": suffix,
            "color": color,
        }

    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
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
                # aim 控制器内部会在早期阶段触发一次性异常（随机偏移导致瞄准误差）
                self._maybe_mark_pick_aim_exception()
            else:
                action = self.place_controller.forward(
                    place_position = state['target_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    gripper_position=state['gripper_position']
                )
            
            # 缓存步骤数据（两个阶段都需要）
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False

        # 控制器完成，检查是否成功
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
            # 任务失败，保存失败数据
            print(f"{self.current_phase.value} task failed!")
            print("保存失败数据...")
            # Mark screenshots after failure with `erro` in filename (no overlay text)
            self._set_frame_exception_once(message="erro", persist_suffix="erro", suffix="")
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
            self._maybe_mark_pick_aim_exception()
        else:
            state['language_instruction'] = self.get_language_instruction()
            action = self.inference_engine.step_inference(state)
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

    def _maybe_mark_pick_aim_exception(self):
        """把 pick_controller_aim 的一次性异常转成 `erro` 文件名标签（不叠字）。"""
        info = getattr(self.pick_controller, "get_aim_error_info", lambda: None)()
        if info is None:
            return
        # 只用 `erro` 作为文件名标签，不在图像上叠字；同时避免 erro 重复
        self._set_frame_exception_once(message="erro", persist_suffix="erro", suffix="", color=info.get("color", (0, 0, 255)))