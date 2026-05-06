from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum
from typing import Optional
from utils.task_utils import TaskUtils
import re

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pour_controller import PourController
from .base_controller import BaseController

class Phase(Enum):
    UNCAPPING = "picking1"
    PLACEING = "placing"
    PICKING = "picking2"
    POURING = "pouring"
    FINISHED = "finished"

class UncapPourController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.every_controller_index = 0
        self.controller_index = 0
        self.initial_bottle_position = None
        self.initial_cap_position = None
        self.initial_targrt_position = None
        
    def _init_collect_mode(self, cfg, robot):
        """Initialize data collection mode"""
        super()._init_collect_mode(cfg, robot)
        
        self.current_phase = TaskPhase.UNCAPPING
        
        # Create RMP controller
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        self.pick_controller1 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 1, 0.05, 0.01, 1]
        )
        
        self.place_controller1 = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        
        self.pick_controller2 = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1]
        )

        self.pour_controller1 = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.008, 0.005, 0.008, 0.5],
            position_threshold=0.02
        )

        self.active_controller = self.pick_controller1
        
    def _set_initial_active_controller(self):
        """Set the initial active controller based on the current phase"""
        controller_map = {
            TaskPhase.UNCAPPING: self.pick_controller1,
            TaskPhase.PLACEING: self.place_controller1,
            TaskPhase.PICKING: self.pick_controller2,
            TaskPhase.POURING: self.pour_controller1,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
        else:
            self.active_controller = self.pick_controller1
        
    def _init_infer_mode(self, cfg, robot):
        """Initialize inference mode"""
        super()._init_infer_mode(cfg, robot)
        
    def reset(self):
        """Reset controller state"""
        super().reset()
        self.initial_bottle_position = None
        self.initial_cap_position = None
        self.initial_targrt_position = None
        self.every_controller_index = 0
        self.current_phase = TaskPhase.UNCAPPING
        if self.mode == "collect":
            self.phase_start_frame = 0
            self.pick_controller1.reset()
            self.pour_controller1.reset()
            self.place_controller1.reset()
            self.pick_controller2.reset()
            self.controller_index = 0
            self.active_controller = self.pick_controller1
        else:
            self.inference_engine.reset()
            
    def _check_phase_success(self, state: Dict[str, Any]) -> bool:
        """Check if the current phase is successfully completed
        
        Args:
            state: Current state dictionary
            
        Returns:
            bool: Whether the current phase is successful
        """
        if self.current_phase == TaskPhase.PICKING:
            # Check if the beaker is picked up
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            if self.initial_beaker_position is not None:
                height_diff = beaker_pos[2] - self.initial_beaker_position[2]
                return height_diff > self.LIFT_HEIGHT_THRESHOLD
            return False
            
        elif self.current_phase == TaskPhase.TRANSPORTING:
            # Check if the beaker is successfully placed on the target platform
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            target_pos = state.get('target_position', np.array([0, 0, 0]))
            distance_to_target = np.linalg.norm(beaker_pos[:2] - target_pos[:2])
            height_close = abs(beaker_pos[2] - target_pos[2]) < 0.1
            return distance_to_target < self.TRANSPORT_SUCCESS_THRESHOLD and height_close
            
        elif self.current_phase == TaskPhase.STIRRING:
            # Check if the stirring is completed (based on the number of stirring steps and the position of the glass rod)
            self.stir_step_count += 1
            beaker_pos = state.get('beaker_position', np.array([0, 0, 0]))
            stir_tool_pos = state.get('stir_tool_position', np.array([0, 0, 0]))
            
            # Check if the glass rod is near the beaker
            distance_to_beaker = np.linalg.norm(stir_tool_pos[:2] - beaker_pos[:2])
            in_beaker = distance_to_beaker < 0.05 and stir_tool_pos[2] < beaker_pos[2] + 0.1
            
            return self.stir_step_count > self.STIR_SUCCESS_STEPS and in_beaker
            
        return False
        
    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current phase
        
        Returns:
            Optional[str]: Language instruction for the current phase
        """
        phase_instructions = {
            TaskPhase.UNCAPPING: "Pick up the cap from the bottle", 
            TaskPhase.PICKING: "Pick up the bottle",
            TaskPhase.POURING: "Pour the contents into the beaker",
            TaskPhase.PLACEING: "Place the cap on the table",
        }
        return phase_instructions.get(self.current_phase, "Complete the laboratory task")
        
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one step of control
        
        Args:
            state: Current state dictionary
            
        Returns:
            Tuple: (action, done, success)
        """
        #self.initial_bottle_position = None
        #self.initial_cap_position = None
        #self.initial_targrt_position = None
        if self.initial_bottle_position is None:
            self.initial_bottle_position = self.object_utils.get_geometry_center(object_path="/World/beaker_05")
        if self.initial_cap_position is None:
            self.initial_cap_position = self.object_utils.get_geometry_center(object_path="/World/BalaoVolumetrico_100mL/BalaoVolumetrico_100mL/Vidro_100mL_Glass_MAT_0")
        if self.initial_beaker_position is None:
            self.initial_beaker_position = self.object_utils.get_geometry_center(object_path="/World/beaker2")
        
        self.every_controller_index += 1
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Step in data collection mode"""
        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        # If the current controller is not completed, continue to execute
        if not self.active_controller.is_done():
            action = self._get_phase_action(state)
            
            # Cache data for training
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
                
            return action, False, False
        else:
            next_phase = self._get_next_phase()
            if next_phase:
                self.current_phase = next_phase
                self._switch_active_controller()
                return None, False, False
            else:
                # All phases completed
                print("All phases completed, task successful!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = TaskPhase.FINISHED
                return None, True, True
            
    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Step in inference mode"""
        state['language_instruction'] = ""
        # Use the inference engine to get the action
        action = self.inference_engine.step_inference(state)
        
        # Check if the task is successful (simplified version, actually may need more complex logic)
        return action, False, self.is_success()
        
    def _get_phase_action(self, state: Dict[str, Any]):
        """Get the corresponding action based on the current phase"""
        if self.current_phase == TaskPhase.UNCAPPING:
            return self.pick_controller1.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker_05"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker_05"),
                object_name="beaker_05",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.1,
                pre_offset_z=0.05,
                after_offset_z=0
            )   
        elif self.current_phase == TaskPhase.PICKING:
            return self.pick_controller2.forward(
                picking_position=self.object_utils.get_geometry_center(object_path="/World/beaker_04"),
                current_joint_positions=state['joint_positions'],
                object_size=self.object_utils.get_object_size(object_path="/World/beaker_04"),
                object_name="beaker_04",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                pre_offset_x=0.07,
                pre_offset_z=0.05,
                after_offset_z=0
            )
        elif self.current_phase == TaskPhase.POURING:
            return self.pour_controller1.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.object_utils.get_object_size(object_path="/World/beaker_05"),
                target_position=np.array([0.32, 0.32, 0.90]),
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
                source_name="beaker_05",
                gripper_position=state['gripper_position'],
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
            )
        elif self.current_phase == TaskPhase.PLACEING:
            return self.place_controller1.forward(
                place_position=self.initial_beaker_position1,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_place_z=0.3,
                place_offset_z=0.02
            )
        return None
        
    def _get_next_phase(self) -> Optional[TaskPhase]:
        phase_sequence = [
            TaskPhase.UNCAPPING,
            TaskPhase.PLACEING,
            TaskPhase.PICKING,
            TaskPhase.POURING
        ]
        
        self.controller_index += 1
        if self.controller_index >= len(phase_sequence):
            self.controller_index = 0
            return None
            
        # Update the current phase and return
        self.current_phase = phase_sequence[self.controller_index]
        return self.current_phase
        
    def _switch_active_controller(self):
        """Switch the active controller based on the current phase"""
        self.every_controller_index = 0
        controller_map = {
            TaskPhase.UNCAPPING: self.pick_controller1,
            TaskPhase.PLACEING: self.Place_controller1,
            TaskPhase.PICKING: self.pick_controller2,
            TaskPhase.POURING: self.pour_controller1,
        }
        
        if self.current_phase in controller_map:
            self.active_controller = controller_map[self.current_phase]
            self.active_controller.reset()
            
    def is_success(self) -> bool:
        return False