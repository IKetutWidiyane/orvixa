"""
Virtual workspace management system.
Handles canvas, layers, and spatial object management.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from utils.logger import get_logger
from utils.config import (
    WORKSPACE_WIDTH, WORKSPACE_HEIGHT,
    INITIAL_ZOOM, MIN_ZOOM, MAX_ZOOM
)
from utils.geometry import clamp


@dataclass
class WorkspaceObject:
    object_id: str
    x: float
    y: float
    width: float
    height: float
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0
    layer: int = 0
    selected: bool = False
    locked: bool = False
    media_type: str = "shape"
    source_path: str = ""

    def get_bounds(self) -> Tuple[float, float, float, float]:
        hw = (self.width * self.scale) / 2
        hh = (self.height * self.scale) / 2
        return (self.x - hw, self.y - hh, self.x + hw, self.y + hh)

    def contains_point(self, px: float, py: float) -> bool:
        x_min, y_min, x_max, y_max = self.get_bounds()
        return x_min <= px <= x_max and y_min <= py <= y_max

    def move(self, dx: float, dy: float):
        self.x += dx
        self.y += dy

    def resize(self, scale_factor: float):
        self.scale = clamp(self.scale * scale_factor, 0.1, 5.0)

    def rotate(self, angle: float):
        self.rotation = (self.rotation + angle) % 360


class VirtualWorkspace:
    """Virtual workspace canvas management."""

    def __init__(self, width: int = WORKSPACE_WIDTH, height: int = WORKSPACE_HEIGHT):
        self.logger = get_logger()

        self.width = width
        self.height = height

        # Camera system (legacy)
        self.camera_x = 0.0
        self.camera_y = 0.0

        # 🔥 FIX: renderer compatibility (IMPORTANT)
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.zoom = INITIAL_ZOOM

        # Objects
        self.objects: List[WorkspaceObject] = []
        self.layer_count = 0
        self.max_layers = 10

        # Canvas
        self.canvas = np.zeros((height, width, 4), dtype=np.uint8)

        # State
        self.selected_object: Optional[WorkspaceObject] = None

        self.logger.info(f"Workspace initialized: {width}x{height}")

    # =========================
    # OBJECT MANAGEMENT
    # =========================
    def add_object(self, obj: WorkspaceObject) -> bool:
        if len(self.objects) >= 100:
            self.logger.warning("Object limit reached")
            return False

        obj.layer = min(self.layer_count, self.max_layers - 1)
        self.objects.append(obj)
        return True

    def remove_object(self, object_id: str) -> bool:
        for i, obj in enumerate(self.objects):
            if obj.object_id == object_id:
                self.objects.pop(i)
                return True
        return False

    def get_object(self, object_id: str) -> Optional[WorkspaceObject]:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    # =========================
    # SELECTION
    # =========================
    def select_object_at(self, px: float, py: float):
        sorted_objects = sorted(self.objects, key=lambda o: o.layer, reverse=True)

        for obj in sorted_objects:
            if obj.contains_point(px, py):
                self.deselect_all()
                obj.selected = True
                self.selected_object = obj
                return obj

        self.deselect_all()
        return None

    def deselect_all(self):
        for obj in self.objects:
            obj.selected = False
        self.selected_object = None

    # =========================
    # CAMERA / VIEW
    # =========================
    def move_camera(self, dx: float, dy: float):
        self.camera_x += dx
        self.camera_y += dy

        # 🔥 sync ke renderer offset
        self.offset_x = self.camera_x
        self.offset_y = self.camera_y

    def set_zoom(self, zoom_level: float):
        self.zoom = clamp(zoom_level, MIN_ZOOM, MAX_ZOOM)

    def zoom_in(self, factor: float = 1.1):
        self.set_zoom(self.zoom * factor)

    def zoom_out(self, factor: float = 0.9):
        self.set_zoom(self.zoom * factor)

    def reset_view(self):
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom = INITIAL_ZOOM

    # =========================
    # COORDINATE SYSTEM
    # =========================
    def world_to_screen(self, x, y, screen_w, screen_h):
        sx = (x - self.camera_x) * self.zoom + screen_w / 2
        sy = (y - self.camera_y) * self.zoom + screen_h / 2
        return int(sx), int(sy)

    def screen_to_world(self, sx, sy, screen_w, screen_h):
        wx = (sx - screen_w / 2) / self.zoom + self.camera_x
        wy = (sy - screen_h / 2) / self.zoom + self.camera_y
        return wx, wy

    # =========================
    # UTIL
    # =========================
    def clear_canvas(self):
        self.canvas.fill(0)

    def get_sorted_objects(self):
        return sorted(self.objects, key=lambda o: o.layer)

    def get_stats(self):
        return {
            "object_count": len(self.objects),
            "camera_x": self.camera_x,
            "camera_y": self.camera_y,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "zoom": self.zoom,
            "selected_object": (
                self.selected_object.object_id
                if self.selected_object else None
            )
        }
