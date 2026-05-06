import numpy as np
from typing import Tuple, Any
from scipy.spatial.transform import Rotation as R
from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.move_controller import MoveController


class PipetteOffsetTaskController(BaseController):
    """
    Controller for pipette/dropper pick-and-drop task with offset.
    Sequence (collect and infer both supported without policy inference):
      1) Pick up the pipette/dropper directly
      2) Move to approach pose above desired drop position (/World/beaker_2)
      3) Descend to drop position
      4) Retract and finish
    """
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        # Build atomic controllers on top of RMP controller from BaseController
        self.pick_ctrl = PickController(
            name="pipette_pick",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006]
        )
        self.move_ctrl = MoveController(
            name="pipette_move",
            cspace_controller=self.rmp_controller,
            position_threshold=0.02,
            orientation_threshold=0.1
        )
        self._phase = 0  # 0: pick pipette; 1: move to approach; 2: descend; 3: retract
        self._approach_pos = None
        self._drop_pos = None
        self._retract_pos = None
        self._pipette_picked = False
        # Failure mode: "pick_fail" or "drop_fail" or "drop_during_move" or None
        self.failure_mode = getattr(cfg.task, "failure_mode", None)
        self._move_step_count = 0  # Counter for movement phase (to trigger drop)
        self._has_dropped = False  # Flag to track if pipette was dropped
        # Exception info consumed by main loop for screenshot tagging/annotation
        self._frame_exception = None

    def consume_frame_exception(self):
        """Return and clear per-frame exception info for screenshot tagging/annotation."""
        info = self._frame_exception
        self._frame_exception = None
        return info

    def _set_frame_exception_once(self, message: str, suffix: str = "erro", persist_suffix: str = "erro", color=(0, 0, 255)):
        """Set exception info once per episode (do not spam)."""
        if self._frame_exception is not None:
            return
        self._frame_exception = {
            "message": message,
            "suffix": suffix,
            "persist_suffix": persist_suffix,
            "color": color,
        }

    def _init_infer_mode(self, cfg, robot=None):
        # Override to avoid requiring an external inference engine for this task
        self.trajectory_controller = None
        self.inference_engine = None

    def reset(self) -> None:
        super().reset()
        self.pick_ctrl.reset()
        self.move_ctrl.reset()
        self._phase = 0
        self._approach_pos = None
        self._drop_pos = None
        self._retract_pos = None
        self._pipette_picked = False
        self._move_step_count = 0
        self._has_dropped = False
        self._frame_exception = None

    def _prepare_targets(self, state: dict):
        if self._approach_pos is not None:
            return
        
        # Get desired drop position from state (based on /World/beaker_2)
        desired = state.get("desired_drop_position")
        if desired is None:
            # Fallback: use drop target position (from /World/beaker_2)
            drop_target_pos = state.get('drop_target_position')
            if drop_target_pos is None:
                print("Warning: No drop target position found for /World/beaker_2, using default")
                desired = np.array([0.25, 0.10, 0.90], dtype=np.float32)
            else:
                # Use beaker_2's center position
                desired = np.array(drop_target_pos, dtype=np.float32)
                # Apply drop_height from config
                desired[2] = float(state.get("drop_height", 0.90))
        else:
            # Use the computed desired_drop_position (already includes offset and drop_height)
            desired = np.array(desired, dtype=np.float32)
        
        approach_h = float(state.get("approach_height", 0.20))
        
        # Failure mode: drop_fail - miss the beaker by dropping at wrong position
        if self.failure_mode == "drop_fail":
            # Offset the drop position to miss the beaker
            desired[0] += 0.08  # Offset to the right (miss the beaker)
            desired[1] += 0.08  # Offset forward (miss the beaker)
        
        # Build approach, drop and retract poses
        # All positions are relative to /World/beaker_2
        self._drop_pos = desired.copy()  # Drop position: beaker_2 center + offset, at drop_height
        self._approach_pos = desired.copy()  # Approach position: above beaker_2
        self._approach_pos[2] = max(self._approach_pos[2] + approach_h, self._approach_pos[2] + 0.15)
        self._retract_pos = self._approach_pos.copy()
        self._retract_pos[2] += 0.1  # Extra lift after drop
        
        # Debug: print target positions
        print(f"Target positions prepared for /World/beaker_2:")
        print(f"  Drop position: {self._drop_pos}")
        print(f"  Approach position: {self._approach_pos}")

    def _ee_orientation_quat(self):
        # Tool down: x,y tilt 0, wrist flipped for vertical approach
        return R.from_euler('xyz', np.radians([0.0, 180.0, 0.0])).as_quat()

    def _step_collect_sequence(self, state: dict) -> Tuple[Any, bool, bool]:
        q = state['joint_positions']
        ee_pos = state['gripper_position']
        ori = self._ee_orientation_quat()

        # Phase 0: Pick up the pipette/dropper
        if self._phase == 0:
            print("Picking up the pipette/dropper")
            # For pipette/dropper, use custom gripper distance and offsets
            # Apply position correction for Xform object (geometry center offset)
            pipette_pos = np.array(state['pipette_position'], dtype=np.float32)
            # Correct the position offset (Xform's geometry center is offset from actual position)
            # Only apply Z offset to grab higher on pipette
            pipette_pos[0] -= 0.01  # X offset correction
            pipette_pos[2] += 0.03  # Z offset correction (grab higher on pipette)

            # Failure mode: pick_fail - move to wrong position (miss the pipette)
            if self.failure_mode == "pick_fail":
                # Move to a position slightly offset from the pipette
                pipette_pos[0] += 0.05  # Offset to the right
                pipette_pos[1] += 0.05  # Offset forward
                # Use wider gripper distance so it doesn't grip properly
                gripper_dist = 0.015  # Too wide to grip
            else:
                gripper_dist = 0.0085  # Moderate grip for thin pipette (increased from 0.002 to prevent ejection)

            action = self.pick_ctrl.forward(
                picking_position=pipette_pos,
                current_joint_positions=q,
                object_name=state['pipette_name'],
                object_size=np.array(state['pipette_size'], dtype=np.float32),
                gripper_control=self.gripper_control,
                gripper_position=ee_pos,
                end_effector_orientation=ori,
                pre_offset_z=0.10,      # Lower pre-approach to get closer
                after_offset_z=0.3,    # Lift after picking (increased from 0.20)
                pre_offset_x=0.0,       # No horizontal offset - approach from directly above
                gripper_distances=gripper_dist
            )
            if self.pick_ctrl.is_done():
                self._phase = 1  # Next: move to approach position above /World/beaker_2
                # In pick_fail mode, pipette is not actually picked
                self._pipette_picked = (self.failure_mode != "pick_fail")
                if self.failure_mode == "pick_fail":
                    # Tag screenshots/annotation as error once simulated pick failure occurs
                    self._set_frame_exception_once(message="erro", suffix="pick_fail", persist_suffix="error")
                # Prepare targets for movement to beaker_2
                self._prepare_targets(state)
            # cache data for dataset
            if 'camera_data' in state and self.mode == "collect":
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            return action, False, False

        # Phase 1: Move to approach position above drop target (/World/beaker_2)
        if self._phase == 1:
            print("Moving to approach position above drop target")
            # Ensure targets are prepared (should have been set in Phase 0, but double-check)
            if self._approach_pos is None:
                self._prepare_targets(state)
            
            # Double-check approach position is set
            if self._approach_pos is None:
                # Fallback: use drop target position with approach height
                drop_target_pos = state.get('drop_target_position')
                if drop_target_pos is None:
                    drop_target_pos = [0.25, 0.10, 0.90]  # Default position
                drop_target_pos = np.array(drop_target_pos, dtype=np.float32)
                approach_h = float(state.get("approach_height", 0.20))
                self._approach_pos = drop_target_pos.copy()
                self._approach_pos[2] = max(self._approach_pos[2] + approach_h, self._approach_pos[2] + 0.15)
                self._drop_pos = drop_target_pos.copy()
                self._drop_pos[2] = float(state.get("drop_height", 0.90))
                self._retract_pos = self._approach_pos.copy()
                self._retract_pos[2] += 0.1
            
            
            # Get movement action first
            action = self.move_ctrl.forward(
                target_position=self._approach_pos,
                current_joint_positions=q,
                gripper_position=ee_pos,
                target_orientation=ori
            )
            
            # Failure mode: drop_during_move - open gripper during movement to simulate drop
            if self.failure_mode == "drop_during_move" and not self._has_dropped:
                self._move_step_count += 1
                # After 30 steps of movement, open gripper to drop pipette
                if self._move_step_count >= 30:
                    # Modify action to open gripper and drop the pipette
                    from omni.isaac.core.utils.types import ArticulationAction
                    # Safely get joint positions
                    if action.joint_positions is not None:
                        action_joints = list(action.joint_positions)
                    else:
                        action_joints = [None] * len(q)
                    
                    if len(action_joints) != len(q):
                        action_joints = [None] * len(q)
                    
                    # Open gripper (joints 7 and 8) to drop pipette
                    action_joints[7] = 0.04  # Open gripper
                    action_joints[8] = 0.04  # Open gripper
                    action = ArticulationAction(joint_positions=action_joints)
                    self._has_dropped = True
                    # Tag screenshots/annotation as erro at the moment of dropping
                    self._set_frame_exception_once(message="erro", suffix="drop_during_move", persist_suffix="erro")
                    # Continue moving but pipette is dropped
                    if 'camera_data' in state and self.mode == "collect":
                        self.data_collector.cache_step(
                            camera_images=state['camera_data'],
                            joint_angles=state['joint_positions'][:-1],
                            language_instruction=self.get_language_instruction()
                        )
                    return action, False, False
            if self.move_ctrl.is_done():
                self._phase = 2  # Next: descend
                self.move_ctrl.reset()  # Reset for next move
            if 'camera_data' in state and self.mode == "collect":
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            return action, False, False

        # Phase 2: Descend to drop position
        if self._phase == 2:
            action = self.move_ctrl.forward(
                target_position=self._drop_pos,
                current_joint_positions=q,
                gripper_position=ee_pos,
                target_orientation=ori
            )
            if self.move_ctrl.is_done():
                # Failure mode: drop_fail - once we reach the (wrong) drop position, tag future screenshots as erro
                if self.failure_mode == "drop_fail":
                    self._set_frame_exception_once(message="erro", suffix="drop_fail", persist_suffix="erro")
                self._phase = 3  # Next: retract
                self.move_ctrl.reset()  # Reset for retract
            if 'camera_data' in state and self.mode == "collect":
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            return action, False, False

        # Phase 3: Retract upward
        if self._phase == 3:
            action = self.move_ctrl.forward(
                target_position=self._retract_pos,
                current_joint_positions=q,
                gripper_position=ee_pos,
                target_orientation=ori
            )
            if self.move_ctrl.is_done():
                # Check if this is a failure mode
                if self.failure_mode in ["pick_fail", "drop_fail", "drop_during_move"]:
                    # Mark as failure
                    self._last_success = False
                else:
                    # Normal success
                    self._last_success = True
                
                if self.mode == "collect":
                    self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self.reset_needed = True
                return None, True, self._last_success
            if 'camera_data' in state and self.mode == "collect":
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    language_instruction=self.get_language_instruction()
                )
            return action, False, False

        return None, False, False

    def step(self, state: dict) -> Tuple[Any, bool, bool]:
        # Use same sequence for collect and infer (no learned policy required)
        return self._step_collect_sequence(state)


