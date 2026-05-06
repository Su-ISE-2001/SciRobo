import numpy as np
from typing import Dict, Any, Optional
from .base_task import BaseTask


class PipetteOffsetTask(BaseTask):
    """
    A task that picks up a pipette (dropper), moves to a drop target with offset, and descends.
    - First picks up the pipette/dropper directly
    - Computes a desired drop position by taking the target prim center and adding a planar offset.
    - Exposes pipette info, desired drop position and offset in the state for the controller.
    Config (expected fields under cfg.task):
      - pipette_path: str -> USD prim path of the pipette/dropper to pick up
      - drop_target_path: str -> USD prim path of the drop target surface/object (e.g., beaker)
      - offset_mode: str -> "fixed" | "random" (default: "fixed")
      - offset_xy: [dx, dy] (meters) when offset_mode == "fixed"
      - offset_xy_range:
          x: [min, max]
          y: [min, max]
        when offset_mode == "random"
      - approach_height: float (meters) height above target to approach
      - drop_height: float (meters) absolute z for dispensing (relative to world)
    """
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.pipette_path: Optional[str] = None
        self.drop_target_path: Optional[str] = None
        self.offset_xy: Optional[np.ndarray] = None
        self.desired_drop_position: Optional[np.ndarray] = None
        self.approach_height: float = float(getattr(self.cfg.task, "approach_height", 0.15))
        self.drop_height: float = float(getattr(self.cfg.task, "drop_height", 0.02))

    def _sample_offset(self) -> np.ndarray:
        mode = getattr(self.cfg.task, "offset_mode", "fixed")
        if mode == "random" and hasattr(self.cfg.task, "offset_xy_range"):
            ox = np.random.uniform(self.cfg.task.offset_xy_range.x[0], self.cfg.task.offset_xy_range.x[1])
            oy = np.random.uniform(self.cfg.task.offset_xy_range.y[0], self.cfg.task.offset_xy_range.y[1])
            return np.array([ox, oy], dtype=np.float32)
        # fallback to fixed
        if hasattr(self.cfg.task, "offset_xy"):
            return np.array(self.cfg.task.offset_xy, dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)

    def reset(self):
        super().reset()
        self.robot.initialize()
        
        # Get pipette and drop target paths
        self.pipette_path = getattr(self.cfg.task, "pipette_path", None)
        self.drop_target_path = getattr(self.cfg.task, "drop_target_path", None)
        
        if not self.pipette_path or not self.drop_target_path:
            # No pipette or target configured → trigger reset loop
            self.reset_needed = True
            return

        # Randomize object positions if obj_paths is configured
        if hasattr(self.cfg.task, 'obj_paths') and self.cfg.task.obj_paths:
            for obj_config in self.cfg.task.obj_paths:
                obj_path = obj_config['path']
                position_range = obj_config['position_range']
                # Randomize position within range
                self.randomize_object_position(obj_path, position_range)

        # Compute target center and offset
        target_center = self.object_utils.get_geometry_center(object_path=self.drop_target_path)
        if target_center is None:
            # Target not found, trigger reset
            self.reset_needed = True
            return
            
        self.offset_xy = self._sample_offset()

        # Desired drop position: add planar offset and set z to configured drop_height
        # Ensure target_center is a proper array
        target_center = np.array(target_center, dtype=np.float32)
        if target_center.ndim == 0:
            # If scalar, create a default position
            target_center = np.array([0.3, 0.0, 0.85], dtype=np.float32)
        
        desired = target_center.copy()
        desired[0] += float(self.offset_xy[0])
        desired[1] += float(self.offset_xy[1])
        desired[2] = float(self.drop_height)
        self.desired_drop_position = desired

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        # Provide basic state plus pipette and drop meta
        state = self.get_basic_state_info(
            additional_info={
                "pipette_path": self.pipette_path,
                "pipette_position": self.object_utils.get_geometry_center(object_path=self.pipette_path)
                    if self.pipette_path else None,
                "pipette_size": self.object_utils.get_object_size(object_path=self.pipette_path)
                    if self.pipette_path else None,
                "pipette_name": self.pipette_path.split("/")[-1] if self.pipette_path else None,
                "drop_target_path": self.drop_target_path,
                "drop_target_position": self.object_utils.get_geometry_center(object_path=self.drop_target_path)
                    if self.drop_target_path else None,
                "drop_offset_xy": self.offset_xy.tolist() if self.offset_xy is not None else [0.0, 0.0],
                "desired_drop_position": self.desired_drop_position.tolist()
                    if self.desired_drop_position is not None else None,
                "approach_height": self.approach_height,
                "drop_height": self.drop_height,
            }
        )
        return state


