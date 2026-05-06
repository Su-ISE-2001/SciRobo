from controllers.atomic_actions.move_controller import MoveController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum
from typing import Optional
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from .atomic_actions.open_controller import OpenController
from .atomic_actions.open_controller_fail_grasp import OpenControllerFailGrasp
from .atomic_actions.pick_controller import PickController
from .atomic_actions.pick_controller_drop import PickControllerDrop
from .atomic_actions.pick_controller_incomplete_close import PickControllerIncompleteClose
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.close_controller import CloseController
from .atomic_actions.press_controller import PressController
from .base_controller import BaseController

class Phase(Enum):
    OPEN_DOOR = "opening_door"
    MOVE_HIGHER = "move_higher"
    PICK_BEAKER = "picking_beaker" 
    PLACE_BEAKER = "placing_beaker"
    PICK_BEAKER3 = "picking_beaker3"
    PLACE_BEAKER3 = "placing_beaker3"
    CLOSE_DOOR = "closing_door"
    PRESS_BUTTON = "pressing_button"
    FINISHED = "finished"

class DeviceOperateController(BaseController):
    """Controller for operating a laboratory device with multiple steps.
    
    Steps:
    1. Open device door
    2. Move higher
    3. Pick up beaker
    4. Place beaker inside device
    5. Close device door
    6. Press device button
    """

    def __init__(self, cfg, robot):
        # Set failure-related configs BEFORE base init (base calls _init_collect_mode)
        self.failure_mode = cfg.task.get("failure_mode", None)
        self.incomplete_open_steps = cfg.task.get("incomplete_open_steps", 120)
        self._open_step_counter = 0
        self._open_exception_reported = False
        # Track whether the current episode has failed (used to skip saving for normal data)
        self._episode_failed = False
        self.pick_controller_fail = None
        self.pick_controller_drop = None
        self.pick_controller_incomplete = None
        self.pick_controller_fail_angle = None
        super().__init__(cfg, robot)
        self.success_steps = set() 
        self.current_phase = Phase.OPEN_DOOR
        self.last_phase = None  
        self.initial_handle_position = None
        self.initial_beaker_position = None
        self.initial_beaker3_position = None
        self.initial_button_position = None
        self._frame_exception = None
            
    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )

        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            furniture_type="door"
        )
        if self.failure_mode == "fail_grasp":
            grasp_offset = cfg.task.get("grasp_offset", 0.03)
            self.open_controller = OpenControllerFailGrasp(
                name="open_controller_fail_grasp",
                cspace_controller=rmp_controller,
                gripper=robot.gripper,
                furniture_type="door",
                grasp_offset=grasp_offset,
            )
        
        # 夹取控制器：可选失败模式（仅作用于第一个烧杯）
        pick_position_offset = cfg.task.get("position_offset", 0.08)
        self.pick_controller_fail = None
        self.pick_controller_drop = None
        self.pick_controller_fail_angle = None
        self.pick_controller_incomplete = None
        if self.failure_mode == "fail_position_pick":
            from controllers.atomic_actions.pick_controller_fail_position import PickControllerFailPosition
            self.pick_controller_fail = PickControllerFailPosition(
                name="pick_controller_fail_position",
                cspace_controller=rmp_controller,
                events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006],
                position_offset=pick_position_offset,
            )
        if self.failure_mode == "drop_pick":
            self.pick_controller_drop = PickControllerDrop(
                name="pick_controller_drop",
                cspace_controller=rmp_controller,
                events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.05, 0.004, 0.006],
                drop_phase=5,
            )
        if self.failure_mode == "fail_angle_pick":
            from controllers.atomic_actions.pick_controller_fail_angle import PickControllerFailAngle
            angle_offset_range = cfg.task.get("angle_offset_range_deg", 45.0)  # 对称范围，正负皆可
            self.pick_controller_fail_angle = PickControllerFailAngle(
                name="pick_controller_fail_angle",
                cspace_controller=rmp_controller,
                events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006],
                angle_offset_range_deg=angle_offset_range,
                deviation_report_threshold_deg=cfg.task.get("angle_report_threshold_deg", 20.0),
                about_gripper_axis=True,
            )
        if self.failure_mode == "incomplete_close_pick":
            self.pick_controller_incomplete = PickControllerIncompleteClose(
                name="pick_controller_incomplete_close",
                cspace_controller=rmp_controller,
                events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006],
                incomplete_close_scale=cfg.task.get("incomplete_close_scale", 1.6),
                extra_gap=cfg.task.get("extra_gap", 0.0),
                gap_report_threshold=cfg.task.get("gap_report_threshold", 0.002),
            )
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        # 根据 failure_mode 选择不同的 place 控制器
        failure_mode = cfg.task.get("failure_mode", None)
        if failure_mode == "fail_position":
            from controllers.atomic_actions.place_controller_fail_position import PlaceControllerFailPosition
            position_offset = cfg.task.get("position_offset", 0.08)  # 默认偏移0.08米
            self.place_controller = PlaceControllerFailPosition(
                name="place_controller",
                cspace_controller=rmp_controller,
                gripper=robot.gripper,
                position_offset=position_offset
            )
        else:
            self.place_controller = PlaceController(
                name="place_controller",
                cspace_controller=rmp_controller,
                gripper=robot.gripper,
            )
        
        self.close_controller = CloseController(
            name="close_controller",
            cspace_controller=rmp_controller,
            furniture_type="door"
        )
        
        self.press_controller = PressController(
            name="press_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.004, 0.05, 0.02],
            initial_offset=0.05,
        )
        
        self.move_controller = MoveController(
            name="move_controller",
            cspace_controller=rmp_controller,
        )
        
        self.active_controller = self.open_controller

    def _init_infer_mode(self, cfg, robot):
        """Initialize controller for inference mode."""
        super()._init_infer_mode(cfg, robot)
        
    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.OPEN_DOOR
        self.last_phase = None
        self.success_steps.clear()
        self._episode_failed = False
        self.initial_handle_position = None  
        self.initial_beaker_position = None
        self.initial_beaker3_position = None
        self.initial_button_position = None
        self.end_handle_position = None
        self._open_step_counter = 0
        self._frame_exception = None
        self._open_exception_reported = False
        
        if self.mode == "collect":
            self.active_controller = self.open_controller
            self.open_controller.reset()
            self.pick_controller.reset()
            if self.pick_controller_fail is not None:
                self.pick_controller_fail.reset()
            if self.pick_controller_drop is not None:
                self.pick_controller_drop.reset()
            if self.pick_controller_fail_angle is not None:
                self.pick_controller_fail_angle.reset()
            if self.pick_controller_incomplete is not None:
                self.pick_controller_incomplete.reset()
            self.place_controller.reset()
            self.close_controller.reset()
            self.press_controller.reset()
            self.move_controller.reset()
        else:
            pass

    def _check_phase_success(self, state):
        """Check success criteria for current phase."""
        # 对于 no_open 模式，开门阶段总是成功（因为已经跳过）
        if self.current_phase == Phase.OPEN_DOOR:
            if self.failure_mode == "no_open":
                return True
            if self.failure_mode == "fail_grasp":
                # 失败模式：夹错把手位置，允许直接通过以进入后续阶段
                return True
            current_pos = state['door_handle_position']
            gripper_position = state['gripper_position']
            if current_pos is None or self.initial_handle_position is None or gripper_position is None:
                return False
            end_effector_distance = abs(np.linalg.norm(np.array(gripper_position) - np.array(current_pos)))
            distance = abs(np.linalg.norm(np.array(current_pos) - self.initial_handle_position))
            return distance > 0.13 and end_effector_distance > 0.04
        elif self.current_phase == Phase.MOVE_HIGHER:
            return state['gripper_position'][2] - self.end_handle_position[2] > 0.15
        elif self.current_phase == Phase.PICK_BEAKER:
            # 对于 no_pick 模式，夹取阶段总是成功（因为已经跳过）
            if self.failure_mode == "no_pick":
                return True
            if self.failure_mode == "fail_position_pick":
                # 位置异常夹取，允许通过以进入下一阶段
                return True
            if self.failure_mode == "drop_pick":
                # 掉落异常夹取，允许通过以进入下一阶段
                return True
            if self.failure_mode == "fail_angle_pick":
                # 角度异常夹取，允许通过以进入下一阶段
                return True
            if self.failure_mode == "incomplete_close_pick":
                # 夹爪未闭合异常，允许通过以进入下一阶段
                return True
            object_pos = state['beaker_position']
            return object_pos[2] > self.initial_beaker_position[2] + 0.05  
        elif self.current_phase == Phase.PLACE_BEAKER:
            object_pos = state['beaker_position']
            target_pos = state['device_interior_position']
            dist = np.linalg.norm(object_pos[:2] - target_pos[:2])
            print(dist, abs(object_pos[2] - target_pos[2]))
            return dist < 0.2 and abs(object_pos[2] - target_pos[2]) < 0.1  
        elif self.current_phase == Phase.PICK_BEAKER3:
            # 对于 no_pick 模式，夹取阶段总是成功（因为已经跳过）
            if self.failure_mode == "no_pick":
                return True
            object_pos = state['beaker3_position']
            return object_pos[2] > self.initial_beaker3_position[2] + 0.05  
        elif self.current_phase == Phase.PLACE_BEAKER3:
            object_pos = state['beaker3_position']
            target_pos = state['beaker3_target_position']
            dist = np.linalg.norm(object_pos[:2] - target_pos[:2])
            return dist < 0.2 and abs(object_pos[2] - target_pos[2]) < 0.1  
        elif self.current_phase == Phase.CLOSE_DOOR:
            # 关门阶段成功判定：既要控制器完成，也要门回到初始（闭合）位置
            if not self.close_controller.is_done():
                return False
            current_pos = state.get('door_handle_position')
            if current_pos is None or self.initial_handle_position is None:
                return False
            
            # 对于门，主要检查 X 方向的距离（门的开关方向）
            # 门关闭时，X 方向应该回到初始位置
            x_distance = abs(np.array(current_pos)[0] - np.array(self.initial_handle_position)[0])
            # 也检查整体距离作为辅助判断
            distance_to_closed = np.linalg.norm(np.array(current_pos) - np.array(self.initial_handle_position))
            
            # 若有门铰链角度信息，再加一重角度约束（接近闭合位置）
            revolute_joint_position = state.get('revolute_joint_position')
            angle_ok = True
            if revolute_joint_position is not None and len(revolute_joint_position) > 0:
                angle_ok = abs(revolute_joint_position[0]) < 0.1  # 放宽角度阈值：从0.05改为0.1（约5.7度）
            
            # 放宽距离阈值：X方向距离<0.05米，整体距离<0.08米，并添加调试信息
            success = x_distance < 0.05 and distance_to_closed < 0.08 and angle_ok
            if not success:
                print(f"[CLOSE_DOOR] 失败: x_distance={x_distance:.4f} (需要<0.05), "
                      f"distance_to_closed={distance_to_closed:.4f} (需要<0.08), "
                      f"angle={abs(revolute_joint_position[0]) if revolute_joint_position is not None and len(revolute_joint_position) > 0 else 'N/A'} (需要<0.1)")
            return success
        elif self.current_phase == Phase.PRESS_BUTTON:
            # 对于 no_press 模式，按按钮阶段总是成功（因为已经跳过）
            if self.failure_mode == "no_press":
                return True
            return state['button_position'][0] - self.initial_button_position[0] > 0.005
        return False

    def _advance_to_next_phase(self):
        """Advance to the next phase of the task."""
        # 根据 failure_mode 调整阶段序列
        if self.failure_mode == "no_open":
            # 跳过开门阶段
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,  # 直接跳过到 MOVE_HIGHER
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "no_press":
            # 跳过按按钮阶段
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.FINISHED,  # 直接完成，跳过按按钮
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "no_pick":
            # 跳过夹取阶段（跳过 PICK_BEAKER 和 PICK_BEAKER3）
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PLACE_BEAKER,  # 跳过 PICK_BEAKER
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,  # 如果进入这个阶段，直接跳到 PLACE_BEAKER
                Phase.PLACE_BEAKER: Phase.PLACE_BEAKER3,  # 跳过 PICK_BEAKER3
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,  # 如果进入这个阶段，直接跳到 PLACE_BEAKER3
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "no_close":
            # 跳过关门阶段（CLOSE_DOOR）
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.PRESS_BUTTON,  # 直接跳到按按钮
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "incomplete_open":
            # 开门阶段提前停止，但流程保持一致
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "fail_grasp":
            # 夹错把手，但流程保持一致
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "fail_position_pick":
            # 夹取位置错误，仅作用于第一个烧杯
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "drop_pick":
            # 夹取过程中掉落，仅作用于第一个烧杯
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "fail_angle_pick":
            # 夹取角度异常，仅作用于第一个烧杯
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        elif self.failure_mode == "incomplete_close_pick":
            # 夹爪未完全闭合，仅作用于第一个烧杯
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        else:
            # 正常流程
            phase_sequence = {
                Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
                Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
                Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
                Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
                Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
                Phase.PLACE_BEAKER3: Phase.CLOSE_DOOR,
                Phase.CLOSE_DOOR: Phase.PRESS_BUTTON,
                Phase.PRESS_BUTTON: Phase.FINISHED
            }
        self.success_steps.add(self.current_phase)

        self.last_phase = self.current_phase
        self.current_phase = phase_sequence.get(self.current_phase, Phase.FINISHED)
        
        # 在 no_pick 模式且 move_higher 完成后，强制后续帧带 erro 后缀
        if self.mode == "collect" and self.failure_mode == "no_pick" and self.last_phase == Phase.MOVE_HIGHER:
            self._frame_exception = {
                "type": "no_pick",
                "message": "no_pick mode active (after move_higher)",
                "suffix": "erro",
                "color": (0, 0, 255),
                "frame_idx": None,
                "episode": self._episode_num,
                "persist_suffix": "erro",
            }
            print(f"[DeviceOperate] no_pick -> set persistent erro after move_higher, episode {self._episode_num}")
        # 在 no_close 模式且 place_beaker3 完成后，强制后续帧带 erro 后缀
        if self.mode == "collect" and self.failure_mode == "no_close" and self.last_phase == Phase.PLACE_BEAKER3:
            self._frame_exception = {
                "type": "no_close",
                "message": "no_close mode active (after placing beaker3)",
                "suffix": "erro",
                "color": (0, 0, 255),
                "frame_idx": None,
                "episode": self._episode_num,
                "persist_suffix": "erro",
            }
            print(f"[DeviceOperate] no_close -> set persistent erro after place_beaker3, episode {self._episode_num}")
        
        if self.mode == "collect":
            controller_map = {
                Phase.OPEN_DOOR: self.open_controller,
                Phase.MOVE_HIGHER: self.move_controller,
                Phase.PICK_BEAKER: (
                    self.pick_controller_drop
                    if self.pick_controller_drop is not None
                    else (
                        self.pick_controller_fail_angle
                        if self.pick_controller_fail_angle is not None
                        else (
                            self.pick_controller_incomplete
                            if self.pick_controller_incomplete is not None
                            else (
                                self.pick_controller_fail
                                if self.pick_controller_fail is not None
                                else self.pick_controller
                            )
                        )
                    )
                ),
                Phase.PLACE_BEAKER: self.place_controller,
                Phase.PICK_BEAKER3: self.pick_controller,
                Phase.PLACE_BEAKER3: self.place_controller,
                Phase.CLOSE_DOOR: self.close_controller,
                Phase.PRESS_BUTTON: self.press_controller
            }
            self.active_controller = controller_map.get(self.current_phase)
            if self.active_controller:
                self.active_controller.reset()

    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        if not hasattr(self, "_failure_mode_logged"):
            print(f"[DeviceOperate] failure_mode = {self.failure_mode}, episode {self._episode_num}")
            self._failure_mode_logged = True
            
        if self.initial_beaker_position is None:
            self.initial_beaker_position = state['beaker_position']
        if self.initial_beaker3_position is None:
            self.initial_beaker3_position = state['beaker3_position']
        if self.initial_button_position is None:
            self.initial_button_position = state['button_position']
            
        if self.current_phase == Phase.FINISHED:
            # 写入数据并检查 episode 数
            if self.mode == "collect":
                if self.data_collector.episode_count >= self.data_collector.max_episodes:
                    print(f"Reached max_episodes ({self.data_collector.max_episodes}), stopping data collection.")
                    self._last_success = False
                    self.reset_needed = True
                    return None, True, False
                # 在正常数据采集时，如果 episode 失败则不写入缓存数据
                should_save = not (self.failure_mode is None and self._episode_failed)
                if should_save:
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                else:
                    print("[DeviceOperate] episode failed in normal mode, skip saving cached data.")
                    self.data_collector.clear_cache()
            self.reset_needed = True
            print(self.success_steps)
            self._last_success = len(self.success_steps) == 8 and not self._episode_failed
            return None, True, self._last_success

        if self.mode == "collect":
            # Collection mode using atomic controllers
            action = None
            
            # 对于 no_open 模式，跳过开门阶段
            if self.current_phase == Phase.OPEN_DOOR:
                if self.failure_mode == "no_open":
                    print("警告：跳过开门阶段！")
                    # 直接标记为成功并进入下一阶段
                    self.success_steps.add(Phase.OPEN_DOOR)
                    if self.initial_handle_position is None:
                        self.initial_handle_position = state['door_handle_position']
                    self.end_handle_position = state['door_handle_position']
                    # 标记异常，用于截图命名
                    self._frame_exception = {
                        "type": "no_open",
                        "message": "Skipped door opening phase",
                        "suffix": "erro",
                        "color": (0, 0, 255),
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                    }
                    self._advance_to_next_phase()
                    return None, False, False
                else:
                    if self.initial_handle_position is None:
                        self.initial_handle_position = state['door_handle_position']
                        
                    if self.failure_mode == "incomplete_open":
                        self._open_step_counter += 1

                    action = self.open_controller.forward(
                        handle_position=state['door_handle_position'],
                        current_joint_positions=state['joint_positions'],
                        gripper_position=state['gripper_position'],
                        revolute_joint_position=state['revolute_joint_position'],
                        end_effector_orientation=euler_angles_to_quat([0, 100, 0], degrees=True, extrinsic=False),
                        angle=100,
                    )
                    self.end_handle_position = state['door_handle_position']
                    # incomplete_open: 达到步数上限后提前结束开门阶段，并标记 erro
                    if self.failure_mode == "incomplete_open" and self._open_step_counter >= self.incomplete_open_steps:
                        self.success_steps.add(Phase.OPEN_DOOR)
                        self._frame_exception = {
                            "type": "incomplete_open",
                            "message": f"incomplete_open triggered at step {self._open_step_counter}",
                            "suffix": "erro",
                            "color": (0, 0, 255),
                            "frame_idx": state.get("frame_idx"),
                            "episode": self._episode_num,
                            "persist_suffix": "erro",
                        }
                        print(f"[DeviceOperate] incomplete_open -> stop early at step {self._open_step_counter}, episode {self._episode_num}, frame {state.get('frame_idx')}")
                        self._advance_to_next_phase()
                        return action, False, False
                    # fail_grasp: 若原子控制器上报异常，标记 erro 并前进
                    if self.failure_mode == "fail_grasp":
                        info = getattr(self.open_controller, "get_fail_grasp_info", lambda: None)()
                        if info and not self._open_exception_reported:
                            self._frame_exception = info
                            # 确保持续 erro
                            self._frame_exception.setdefault("persist_suffix", "erro")
                            self._frame_exception.setdefault("suffix", "erro")
                            self.success_steps.add(Phase.OPEN_DOOR)
                            self._open_exception_reported = True
                            print(f"[DeviceOperate] fail_grasp -> mark erro, episode {self._episode_num}, frame {state.get('frame_idx')}")
                            self._advance_to_next_phase()
                            return action, False, False
                
            elif self.current_phase == Phase.MOVE_HIGHER:
                target_position = self.end_handle_position.copy()
                target_position[0] -= 0.4
                target_position[2] += 0.22
                action = self.move_controller.forward(
                    target_position=target_position,
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    target_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                )
            elif self.current_phase == Phase.PICK_BEAKER:
                # 对于 no_pick 模式，跳过夹取阶段
                if self.failure_mode == "no_pick":
                    print("警告：跳过夹取第一个烧杯阶段！")
                    # 直接标记为成功并进入下一阶段
                    self.success_steps.add(Phase.PICK_BEAKER)
                    # 标记异常，用于截图命名
                    # 记录当前帧异常，并将持久后缀设为 erro（主循环读取后会持续带上）
                    self._frame_exception = {
                        "type": "no_pick",
                        "message": "Skipped picking first beaker phase",
                        "suffix": "erro",
                        "color": (0, 0, 255),
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                        "persist_suffix": "erro",  # 提示主循环持续追加 erro
                    }
                    print(f"[DeviceOperate] no_pick (first beaker) -> mark erro, episode {self._episode_num}, frame {state.get('frame_idx')}")
                    self._advance_to_next_phase()
                    return None, False, False
                elif self.failure_mode == "fail_position_pick":
                    controller = self.pick_controller_fail
                    action = controller.forward(
                        picking_position=state['beaker_position'],
                        current_joint_positions=state['joint_positions'],
                        object_size=state['beaker_size'],
                        object_name="beaker",
                        gripper_control=self.gripper_control,
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    )
                    info = controller.get_position_error_info()
                    if info is not None:
                        info["frame_idx"] = state.get("frame_idx")
                        info["episode"] = self._episode_num
                        # 保持错误后缀
                        info.setdefault("suffix", "erro")
                        info.setdefault("persist_suffix", "erro")
                        self._frame_exception = info
                elif self.failure_mode == "drop_pick":
                    controller = self.pick_controller_drop
                    action = controller.forward(
                        picking_position=state['beaker_position'],
                        current_joint_positions=state['joint_positions'],
                        object_size=state['beaker_size'],
                        object_name="beaker",
                        gripper_control=self.gripper_control,
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    )
                    info = controller.get_drop_info()
                    if info is not None:
                        info["frame_idx"] = state.get("frame_idx")
                        info["episode"] = self._episode_num
                        info.setdefault("suffix", "erro")
                        info.setdefault("persist_suffix", "erro")
                        self._frame_exception = info
                elif self.failure_mode == "fail_angle_pick":
                    controller = self.pick_controller_fail_angle
                    action = controller.forward(
                        picking_position=state['beaker_position'],
                        current_joint_positions=state['joint_positions'],
                        object_size=state['beaker_size'],
                        object_name="beaker",
                        gripper_control=self.gripper_control,
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    )
                    info = controller.get_angle_deviation_info()
                    if info is not None:
                        info["frame_idx"] = state.get("frame_idx")
                        info["episode"] = self._episode_num
                        info.setdefault("suffix", "erro")
                        info.setdefault("persist_suffix", "erro")
                        self._frame_exception = info
                elif self.failure_mode == "incomplete_close_pick":
                    controller = self.pick_controller_incomplete
                    action = controller.forward(
                        picking_position=state['beaker_position'],
                        current_joint_positions=state['joint_positions'],
                        object_size=state['beaker_size'],
                        object_name="beaker",
                        gripper_control=self.gripper_control,
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    )
                    info = controller.get_incomplete_close_info()
                    if info is not None:
                        info["frame_idx"] = state.get("frame_idx")
                        info["episode"] = self._episode_num
                        info.setdefault("suffix", "erro")
                        info.setdefault("persist_suffix", "erro")
                        self._frame_exception = info
                else:
                    action = self.pick_controller.forward(
                        picking_position=state['beaker_position'],
                        current_joint_positions=state['joint_positions'],
                        object_size=state['beaker_size'],
                        object_name="beaker",
                        gripper_control=self.gripper_control,
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    )
            elif self.current_phase == Phase.PLACE_BEAKER:
                action = self.place_controller.forward(
                    place_position=np.array(state['device_interior_position']),
                    current_joint_positions=state['joint_positions'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    gripper_position=state['gripper_position'],
                    gripper_control=self.gripper_control,
                    pre_place_z=0.1,
                )
                if self.failure_mode == "fail_position":
                    info = getattr(self.place_controller, "get_place_position_error_info", lambda: None)()
                    if info is not None:
                        info["frame_idx"] = state.get("frame_idx")
                        info["episode"] = self._episode_num
                        info.setdefault("suffix", "erro")
                        info.setdefault("persist_suffix", "erro")
                        self._frame_exception = info
            elif self.current_phase == Phase.PICK_BEAKER3:
                # 对于 no_pick 模式，跳过夹取阶段
                if self.failure_mode == "no_pick":
                    print("警告：跳过夹取第二个烧杯阶段！")
                    # 直接标记为成功并进入下一阶段
                    self.success_steps.add(Phase.PICK_BEAKER3)
                    # 标记异常，用于截图命名
                    self._frame_exception = {
                        "type": "no_pick",
                        "message": "Skipped picking second beaker phase",
                        "suffix": "erro",
                        "color": (0, 0, 255),
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                        "persist_suffix": "erro",  # 提示主循环持续追加 erro
                    }
                    print(f"[DeviceOperate] no_pick (second beaker) -> mark erro, episode {self._episode_num}, frame {state.get('frame_idx')}")
                    self._advance_to_next_phase()
                    return None, False, False
                else:
                    action = self.pick_controller.forward(
                        picking_position=state['beaker3_position'],
                        current_joint_positions=state['joint_positions'],
                        object_size=state['beaker3_size'],
                        object_name="beaker",
                        gripper_control=self.gripper_control,
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    )
            elif self.current_phase == Phase.PLACE_BEAKER3:
                action = self.place_controller.forward(
                    place_position=np.array(state['beaker3_target_position']),
                    current_joint_positions=state['joint_positions'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 15])).as_quat(),
                    gripper_position=state['gripper_position'],
                    gripper_control=self.gripper_control,
                    pre_place_z=0.15,
                )
                if self.failure_mode == "fail_position":
                    info = getattr(self.place_controller, "get_place_position_error_info", lambda: None)()
                    if info is not None:
                        info["frame_idx"] = state.get("frame_idx")
                        info["episode"] = self._episode_num
                        info.setdefault("suffix", "erro")
                        info.setdefault("persist_suffix", "erro")
                        self._frame_exception = info
            elif self.current_phase == Phase.CLOSE_DOOR:
                # 对于 no_close 模式，跳过关门阶段
                if self.failure_mode == "no_close":
                    print("警告：跳过关门阶段！")
                    self.success_steps.add(Phase.CLOSE_DOOR)
                    self._frame_exception = {
                        "type": "no_close",
                        "message": "Skipped closing door phase",
                        "suffix": "erro",
                        "color": (0, 0, 255),
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                        "persist_suffix": "erro",
                    }
                    print(f"[DeviceOperate] no_close -> mark erro, episode {self._episode_num}, frame {state.get('frame_idx')}")
                    self._advance_to_next_phase()
                    return None, False, False
                else:
                    action = self.close_controller.forward(
                        handle_position=state['door_handle_position'],
                        current_joint_positions=state['joint_positions'],
                        revolute_joint_position=state['revolute_joint_position'],
                        gripper_position=state['gripper_position'],
                        end_effector_orientation=R.from_euler('xyz', np.radians([350, 90, 25])).as_quat(),
                    )
            elif self.current_phase == Phase.PRESS_BUTTON:
                # 对于 no_press 模式，跳过按按钮阶段
                if self.failure_mode == "no_press":
                    print("警告：跳过按按钮阶段！")
                    # 直接标记为成功并完成
                    self.success_steps.add(Phase.PRESS_BUTTON)
                    # 标记异常，用于截图命名，并让后续帧持续带 erro
                    self._frame_exception = {
                        "type": "no_press",
                        "message": "Skipped pressing button phase",
                        "suffix": "erro",
                        "color": (0, 0, 255),
                        "frame_idx": state.get("frame_idx"),
                        "episode": self._episode_num,
                        "persist_suffix": "erro",
                    }
                    print(f"[DeviceOperate] no_press -> mark erro, episode {self._episode_num}, frame {state.get('frame_idx')}")
                    self.current_phase = Phase.FINISHED
                    return None, True, False
                else:
                    action = self.press_controller.forward(
                        target_position=state['button_position'],
                        current_joint_positions=state['joint_positions'],
                        gripper_control=self.gripper_control,
                        end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                        press_distance = 0.005
                    )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1]
                )
            
            if self.active_controller.is_done():
                success = self._check_phase_success(state)
                if success:
                    print(f"{self.current_phase.value} success!")
                    self._advance_to_next_phase()
                    return None, False, False
                else:
                    print(f"{self.current_phase.value} failed! 保存失败数据...")
                    # 正常数据采集时不保存失败数据
                    if self.failure_mode is None:
                        self._episode_failed = True
                        self.data_collector.clear_cache()
                        print("[DeviceOperate] normal mode failure -> cleared cached data, no save.")
                    else:
                        # 失败模式下依然保存失败数据用于训练
                        self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    self.current_phase = Phase.FINISHED
                    return None, True, False
            
            return action, False, False

        else:
            return self._step_infer(state)

    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Uses policy to process observations and generate actions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if self.initial_beaker_position is None:
            self.initial_beaker_position = state['beaker_position']
        if self.initial_beaker3_position is None:
            self.initial_beaker3_position = state['beaker3_position']
        if self.initial_button_position is None:
            self.initial_button_position = state['button_position']
        if self.initial_handle_position is None:
            self.initial_handle_position = state['door_handle_position']

        if self.current_phase != Phase.FINISHED:
            success = self._check_phase_success(state)
            if success:
                print(f"Inference: {self.current_phase.value} success!")
                self.success_steps.add(self.current_phase)
                self._advance_to_next_phase()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            self._last_success = len(self.success_steps) == 7
            return None, True, self._last_success

        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Operate the laboratory device by opening the door, placing beakers, and pressing the button"

        action = self.inference_engine.step_inference(state)
        
        return action, False, False

    def is_success(self):
        """Check if task was successful."""
        return len(self.success_steps) == 7

    def consume_frame_exception(self):
        """供外部读取并清空当前帧异常标记。
        
        正常任务（failure_mode 为空）时不返回异常，用于避免正常数据截屏带异常后缀。
        """
        if self.failure_mode is None:
            self._frame_exception = None
            return None
        info = self._frame_exception
        self._frame_exception = None
        return info

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        self._language_instruction = "First, open the device door by pulling the handle. Then, move the robot arm higher to avoid obstacles. Next, pick up the first beaker from the table and place it inside the device. After that, pick up the second beaker and place it in its designated position. Finally, press the device button to activate the operation."
        return self._language_instruction
