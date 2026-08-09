"""
Shape manager for gesture-drawn shapes in ORVIXA.
Handles shape type selection and creation of shapes from two-finger gestures.
"""

import numpy as np
import time
import uuid
from enum import Enum
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field

from utils.logger import get_logger
from core.workspace.manager import WorkspaceObject


class ShapeType(Enum):
    """Available shape types that can be drawn with two fingers."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    STAR = "star"
    HEXAGON = "hexagon"


@dataclass
class ShapeDrawing:
    """Tracks an in-progress two-finger shape drawing."""
    start_point: Tuple[float, float]
    start_time: float
    shape_type: ShapeType
    hand_id: int
    points: List[Tuple[float, float]] = field(default_factory=list)


class ShapeManager:
    """Manages shape creation from two-finger gestures."""

    def __init__(self):
        """Initialize shape manager."""
        self.logger = get_logger()
        self.active_shape: Optional[ShapeDrawing] = None
        self._drag_threshold = 0.05  # Minimum distance to start drawing (normalized)
        self.logger.info("Shape manager initialized")

    def process_two_fingers(
        self,
        hand_id: int,
        index_pos: Optional[Tuple[float, float]],
        middle_pos: Optional[Tuple[float, float]],
        default_shape: ShapeType = ShapeType.RECTANGLE
    ) -> Optional[WorkspaceObject]:
        """
        Process two-finger gesture to create or update a shape.

        Handles the lifecycle:
        - Start: First call with two fingers (creates start point)
        - Update: While fingers move (extends the shape corners)
        - End: When fingers are released (returns the completed shape object)

        Args:
            hand_id: ID of the hand performing the gesture.
            index_pos: Position of index fingertip (normalized 0-1).
            middle_pos: Position of middle fingertip (normalized 0-1).
            default_shape: Shape type to create.

        Returns:
            Completed WorkspaceObject if drawing finished, otherwise None.
        """
        if index_pos is None or middle_pos is None:
            return self._complete_drawing()

        # Use the midpoint between two fingers as the drawing anchor
        current_pos = (
            (index_pos[0] + middle_pos[0]) / 2.0,
            (index_pos[1] + middle_pos[1]) / 2.0
        )

        if self.active_shape is None:
            # Start new drawing
            self.active_shape = ShapeDrawing(
                start_point=current_pos,
                start_time=time.time(),
                shape_type=default_shape,
                hand_id=hand_id,
                points=[current_pos]
            )
            return None

        # Check if drawing is from the same hand
        if self.active_shape.hand_id != hand_id:
            return self._complete_drawing()

        # Append current position
        self.active_shape.points.append(current_pos)

        # Check if drawing should complete (fingers moved away significantly)
        dx = current_pos[0] - self.active_shape.start_point[0]
        dy = current_pos[1] - self.active_shape.start_point[1]
        distance = np.sqrt(dx * dx + dy * dy)

        # Complete if fingers moved a significant distance (drawing finished)
        if distance > self._drag_threshold * 5:
            return self._complete_drawing()

        return None

    def _complete_drawing(self) -> Optional[WorkspaceObject]:
        """Finalize the shape drawing and create a workspace object."""
        if self.active_shape is None:
            return None

        drawing = self.active_shape
        self.active_shape = None

        # Calculate shape dimensions from start and end points
        if len(drawing.points) < 5:
            return None

        # Find the farthest point from the start
        farthest_dist = 0
        end_point = drawing.points[-1]
        for point in drawing.points:
            dx = point[0] - drawing.start_point[0]
            dy = point[1] - drawing.start_point[1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > farthest_dist:
                farthest_dist = dist
                end_point = point

        dx = end_point[0] - drawing.start_point[0]
        dy = end_point[1] - drawing.start_point[1]
        distance = np.sqrt(dx * dx + dy * dy)

        # Require minimum size for the shape
        min_size = 0.05
        if distance < min_size:
            return None

        # Convert to workspace coordinates (center point and size)
        center_x = (drawing.start_point[0] + end_point[0]) / 2.0
        center_y = (drawing.start_point[1] + end_point[1]) / 2.0

        # Compute size in workspace units (simplified: use zoom 1.0 conversion)
        # Workspace is 4000x3000, converted from normalized 0-1
        workspace_w = 4000.0
        workspace_h = 3000.0
        world_cx = center_x * workspace_w
        world_cy = center_y * workspace_h
        world_w = abs(dx) * workspace_w
        world_h = abs(dy) * workspace_h

        # Apply rotation based on gesture slope
        angle = np.degrees(np.arctan2(dy, dx))

        # Create workspace object
        object_id = f"shape_{uuid.uuid4().hex[:8]}"
        shape_obj = WorkspaceObject(
            object_id=object_id,
            x=world_cx,
            y=world_cy,
            width=max(world_w, 50),
            height=max(world_h, 50),
            rotation=angle,
            media_type=f"shape_{drawing.shape_type.value}"
        )

        self.logger.info(f"Created shape: {drawing.shape_type.value} "
                         f"size={world_w:.0f}x{world_h:.0f} at ({world_cx:.0f},{world_cy:.0f})")
        return shape_obj

    def cancel_drawing(self):
        """Cancel any in-progress shape drawing."""
        self.active_shape = None

    def is_drawing(self) -> bool:
        """Check if a shape drawing is currently active."""
        return self.active_shape is not None