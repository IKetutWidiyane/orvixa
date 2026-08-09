"""
Rendering engine for ORVIXA.
Handles futuristic AR rendering, webcam overlay,
workspace visuals, hologram effects, and HUD rendering.
"""

import cv2
import numpy as np
import math

from utils.logger import get_logger
from utils.config import COLORS
from core.workspace.manager import WorkspaceObject, VirtualWorkspace
from core.shapes.shape_manager import ShapeType


class Renderer:
    def __init__(self, width: int, height: int):
        self.logger = get_logger()
        self.width = width
        self.height = height
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.grid_size = 50
        self.grid_color = (30, 30, 40)
        self.hologram_scan_offset = 0
        self._vignette_mask = None
        self._vignette_shape = None
        self.logger.info(f"Renderer initialized: {width}x{height}")

    def set_background(self, background_frame: np.ndarray):
        if background_frame is None:
            return
        self.frame = background_frame.copy()
        self.height, self.width = self.frame.shape[:2]
        self._ensure_vignette_mask()

    def clear(self):
        self.frame[:] = COLORS.get("dark_bg", (0, 0, 0))

    def render_workspace_background(self, workspace: VirtualWorkspace):
        if self.frame is None:
            return

        overlay = self.frame.copy()
        offset_x = int(getattr(workspace, "offset_x", 0) or 0) % self.grid_size
        offset_y = int(getattr(workspace, "offset_y", 0) or 0) % self.grid_size

        for x in range(-offset_x, self.width, self.grid_size):
            cv2.line(overlay, (x, 0), (x, self.height), self.grid_color, 1)
        for y in range(-offset_y, self.height, self.grid_size):
            cv2.line(overlay, (0, y), (self.width, y), self.grid_color, 1)

        self.frame = cv2.addWeighted(overlay, 0.18, self.frame, 0.82, 0)

    def render_object(self, obj: WorkspaceObject, workspace: VirtualWorkspace):
        screen_x, screen_y = workspace.world_to_screen(
            obj.x, obj.y, self.width, self.height
        )
        screen_w = int(obj.width * obj.scale * workspace.zoom)
        screen_h = int(obj.height * obj.scale * workspace.zoom)
        if screen_w <= 0 or screen_h <= 0:
            return

        media_frame = self._get_object_frame(obj, screen_w, screen_h)
        if media_frame is not None:
            self._draw_media_frame(media_frame, screen_x, screen_y, obj.opacity)
        elif obj.media_type.startswith("shape_"):
            self._draw_shape_object(obj, screen_x, screen_y, screen_w, screen_h)
        else:
            cv2.rectangle(
                self.frame,
                (screen_x - screen_w // 2, screen_y - screen_h // 2),
                (screen_x + screen_w // 2, screen_y + screen_h // 2),
                COLORS["neon_blue"],
                2,
            )

        if obj.selected:
            self._draw_selection_box(screen_x, screen_y, screen_w, screen_h)
        self._draw_layer_indicator(screen_x, screen_y, obj.layer)

    def _draw_shape_object(self, obj: WorkspaceObject, cx: int, cy: int, w: int, h: int):
        """Draw a shape object with neon hologram style."""
        shape_name = obj.media_type.replace("shape_", "")
        color = COLORS["neon_blue"]
        fill_color = (15, 30, 50)  # Subtle fill

        # Build points for polygon shapes
        hw, hh = w // 2, h // 2
        points = []

        if shape_name == "rectangle":
            x1, y1 = cx - hw, cy - hh
            x2, y2 = cx + hw, cy + hh
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), fill_color, -1)
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 3)
            self._draw_shape_glow(self.frame, cx, cy, w, h, color)

        elif shape_name == "circle":
            radius = min(hw, hh)
            cv2.circle(self.frame, (cx, cy), radius, fill_color, -1)
            cv2.circle(self.frame, (cx, cy), radius, color, 3)
            self._draw_shape_glow(self.frame, cx, cy, radius * 2, radius * 2, color)

        elif shape_name == "triangle":
            points = [
                (cx, cy - hh),
                (cx - hw, cy + hh),
                (cx + hw, cy + hh),
            ]
            self._fill_polygon(points, fill_color)
            cv2.polylines(self.frame, [np.array(points)], True, color, 3)
            self._draw_shape_glow(self.frame, cx, cy, w, h, color)

        elif shape_name == "diamond":
            points = [
                (cx, cy - hh),
                (cx + hw, cy),
                (cx, cy + hh),
                (cx - hw, cy),
            ]
            self._fill_polygon(points, fill_color)
            cv2.polylines(self.frame, [np.array(points)], True, color, 3)
            self._draw_shape_glow(self.frame, cx, cy, w, h, color)

        elif shape_name == "star":
            outer_radius = min(hw, hh)
            inner_radius = outer_radius * 0.4
            center = (cx, cy)
            points = []
            for i in range(10):
                radius = outer_radius if i % 2 == 0 else inner_radius
                angle = math.radians(-90 + i * 36)
                px = center[0] + radius * math.cos(angle)
                py = center[1] + radius * math.sin(angle)
                points.append((int(px), int(py)))
            self._fill_polygon(points, fill_color)
            cv2.polylines(self.frame, [np.array(points)], True, color, 3)
            self._draw_shape_glow(self.frame, cx, cy, outer_radius * 2, outer_radius * 2, color)

        elif shape_name == "hexagon":
            radius = min(hw, hh)
            center = (cx, cy)
            points = []
            for i in range(6):
                angle = math.radians(60 * i)
                px = center[0] + radius * math.cos(angle)
                py = center[1] + radius * math.sin(angle)
                points.append((int(px), int(py)))
            self._fill_polygon(points, fill_color)
            cv2.polylines(self.frame, [np.array(points)], True, color, 3)
            self._draw_shape_glow(self.frame, cx, cy, radius * 2, radius * 2, color)

    def _fill_polygon(self, points: list, color):
        """Fill a polygon with given color."""
        if len(points) < 3:
            return
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(self.frame, [pts], color)

    def _draw_shape_glow(self, frame, cx, cy, w, h, color):
        """Draw a subtle glow effect around shapes."""
        glow_frame = frame.copy()
        # Outer glow
        if w > 2 and h > 2:
            cv2.rectangle(glow_frame, (cx - w // 2 - 8, cy - h // 2 - 8),
                          (cx + w // 2 + 8, cy + h // 2 + 8),
                          color, 1)
        frame[:] = cv2.addWeighted(glow_frame, 0.25, frame, 0.75, 0)

    def _get_object_frame(self, obj: WorkspaceObject, width: int, height: int):
        frame = None
        if hasattr(obj, "get_display_frame"):
            frame = obj.get_display_frame()
        elif hasattr(obj, "get_display_image"):
            frame = obj.get_display_image()
        if frame is None:
            return None

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        rotation = getattr(obj, "rotation", 0.0) or 0.0
        if abs(rotation) > 0.01:
            center = (width / 2, height / 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            frame = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        return frame

    def _draw_media_frame(self, media_frame: np.ndarray, cx: int, cy: int, opacity: float):
        h, w = media_frame.shape[:2]
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h

        frame_x1 = max(0, x1)
        frame_y1 = max(0, y1)
        frame_x2 = min(self.width, x2)
        frame_y2 = min(self.height, y2)
        if frame_x1 >= frame_x2 or frame_y1 >= frame_y2:
            return

        media_x1 = frame_x1 - x1
        media_y1 = frame_y1 - y1
        media_x2 = media_x1 + (frame_x2 - frame_x1)
        media_y2 = media_y1 + (frame_y2 - frame_y1)

        roi = self.frame[frame_y1:frame_y2, frame_x1:frame_x2]
        src = media_frame[media_y1:media_y2, media_x1:media_x2]
        alpha = max(0.0, min(1.0, float(opacity)))

        if src.shape[2] == 4:
            src_alpha = (src[:, :, 3:4].astype(np.float32) / 255.0) * alpha
            blended = src[:, :, :3].astype(np.float32) * src_alpha + roi.astype(np.float32) * (1.0 - src_alpha)
            roi[:] = blended.astype(np.uint8)
        else:
            cv2.addWeighted(src, alpha, roi, 1.0 - alpha, 0, roi)

    def render_hand(self, landmarks: np.ndarray, color=None):
        if landmarks is None:
            return
        if color is None:
            color = COLORS["neon_blue"]

        h, w = self.frame.shape[:2]
        pixel_landmarks = (landmarks[:, :2] * np.array([w, h])).astype(int)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        for s, e in connections:
            if s < len(pixel_landmarks) and e < len(pixel_landmarks):
                cv2.line(self.frame, tuple(pixel_landmarks[s]), tuple(pixel_landmarks[e]), color, 2)

        for i, lm in enumerate(pixel_landmarks):
            cv2.circle(self.frame, tuple(lm), 6 if i in [4, 8, 12, 16, 20] else 4, color, -1)

    def render_cursor(self, x: int, y: int, size: int = 20):
        if x is None or y is None:
            return
        color = COLORS["neon_blue"]
        cv2.circle(self.frame, (x, y), size, color, 2)
        cv2.line(self.frame, (x - size, y), (x + size, y), color, 1)
        cv2.line(self.frame, (x, y - size), (x, y + size), color, 1)
        cv2.circle(self.frame, (x, y), 4, color, -1)

    def render_text(self, text, x, y, color=None, font_scale=0.8, thickness=1):
        if color is None:
            color = COLORS["neon_blue"]
        cv2.putText(self.frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(self.frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def render_hologram_scanlines(self):
        if self.frame is None:
            return
        self.frame[::6] = (self.frame[::6].astype(np.float32) * 0.75).astype(np.uint8)

    def apply_vignette(self):
        if self.frame is None:
            return
        self._ensure_vignette_mask()
        if self._vignette_mask is None:
            return
        self.frame = (self.frame.astype(np.float32) * self._vignette_mask).astype(np.uint8)

    def _draw_selection_box(self, cx, cy, w, h):
        color = COLORS["neon_blue"]
        corner = 20
        ox = w // 2
        oy = h // 2
        points = [
            ((cx - ox, cy - oy), (cx - ox + corner, cy - oy)),
            ((cx - ox, cy - oy), (cx - ox, cy - oy + corner)),
            ((cx + ox, cy - oy), (cx + ox - corner, cy - oy)),
            ((cx + ox, cy - oy), (cx + ox, cy - oy + corner)),
            ((cx - ox, cy + oy), (cx - ox + corner, cy + oy)),
            ((cx - ox, cy + oy), (cx - ox, cy + oy - corner)),
            ((cx + ox, cy + oy), (cx + ox - corner, cy + oy)),
            ((cx + ox, cy + oy), (cx + ox, cy + oy - corner)),
        ]
        for p1, p2 in points:
            cv2.line(self.frame, p1, p2, color, 2)

    def _draw_layer_indicator(self, cx, cy, layer):
        cv2.putText(
            self.frame,
            f"L{layer}",
            (cx + 20, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS["purple"],
            1,
        )

    def get_frame(self):
        self.render_hologram_scanlines()
        self.apply_vignette()
        return self.frame.copy()

    def _ensure_vignette_mask(self):
        shape = (self.height, self.width)
        if self._vignette_shape == shape and self._vignette_mask is not None:
            return

        kernel_x = cv2.getGaussianKernel(self.width, 250)
        kernel_y = cv2.getGaussianKernel(self.height, 250)
        mask = kernel_y * kernel_x.T
        mask = mask / mask.max()
        self._vignette_mask = np.dstack([mask, mask, mask]).astype(np.float32)
        self._vignette_shape = shape
