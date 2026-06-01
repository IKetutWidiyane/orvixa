"""
Rendering engine for ORVIXA.
Handles futuristic AR rendering, webcam overlay,
workspace visuals, hologram effects, and HUD rendering.
"""

import cv2
import numpy as np
from typing import Tuple

from utils.logger import get_logger
from utils.config import COLORS
from utils.geometry import (
    draw_circle_with_glow,
    draw_line_with_glow
)

from core.workspace.manager import (
    WorkspaceObject,
    VirtualWorkspace
)


class Renderer:

    def __init__(self, width: int, height: int):
        self.logger = get_logger()

        self.width = width
        self.height = height

        self.frame = np.zeros((height, width, 3), dtype=np.uint8)

        self.grid_size = 50
        self.grid_color = (30, 30, 40)

        # 🔥 safety for hologram system
        self.hologram_scan_offset = 0

        self.logger.info(f"Renderer initialized: {width}x{height}")

    # =========================
    # FRAME
    # =========================
    def set_background(self, background_frame: np.ndarray):
        if background_frame is None:
            return
        self.frame = background_frame.copy()
        self.height, self.width = self.frame.shape[:2]

    def clear(self):
        self.frame[:] = COLORS.get('dark_bg', (0, 0, 0))

    # =========================
    # WORKSPACE GRID
    # =========================
    def render_workspace_background(self, workspace: VirtualWorkspace):

        if self.frame is None:
            return

        overlay = self.frame.copy()

        offset_x = int(getattr(workspace, "offset_x", 0) or 0)
        offset_y = int(getattr(workspace, "offset_y", 0) or 0)

        offset_x %= self.grid_size
        offset_y %= self.grid_size

        # vertical lines
        for x in range(-offset_x, self.width, self.grid_size):
            cv2.line(overlay, (x, 0), (x, self.height), self.grid_color, 1)

        # horizontal lines
        for y in range(-offset_y, self.height, self.grid_size):
            cv2.line(overlay, (0, y), (self.width, y), self.grid_color, 1)

        self.frame = cv2.addWeighted(overlay, 0.18, self.frame, 0.82, 0)

    # =========================
    # OBJECT
    # =========================
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
        else:
            cv2.rectangle(
                self.frame,
                (screen_x - screen_w // 2, screen_y - screen_h // 2),
                (screen_x + screen_w // 2, screen_y + screen_h // 2),
                COLORS['neon_blue'],
                2
            )

        if obj.selected:
            self._draw_selection_box(screen_x, screen_y, screen_w, screen_h)

        self._draw_layer_indicator(screen_x, screen_y, obj.layer)

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
                borderValue=(0, 0, 0)
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

    # =========================
    # HAND
    # =========================
    def render_hand(self, landmarks: np.ndarray, color=None):

        if landmarks is None:
            return

        if color is None:
            color = COLORS['neon_blue']

        h, w = self.frame.shape[:2]

        pixel_landmarks = (landmarks[:, :2] * np.array([w, h])).astype(int)

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        for s, e in connections:
            if s < len(pixel_landmarks) and e < len(pixel_landmarks):
                cv2.line(self.frame,
                         tuple(pixel_landmarks[s]),
                         tuple(pixel_landmarks[e]),
                         color, 2)

        for i, lm in enumerate(pixel_landmarks):
            if lm is not None:
                cv2.circle(
                    self.frame,
                    tuple(lm),
                    6 if i in [4, 8, 12, 16, 20] else 4,
                    color,
                    -1
                )

    # =========================
    # CURSOR
    # =========================
    def render_cursor(self, x: int, y: int, size: int = 20):

        if x is None or y is None:
            return

        color = COLORS['neon_blue']

        cv2.circle(self.frame, (x, y), size, color, 2)
        cv2.line(self.frame, (x - size, y), (x + size, y), color, 1)
        cv2.line(self.frame, (x, y - size), (x, y + size), color, 1)
        cv2.circle(self.frame, (x, y), 4, color, -1)

    # =========================
    # TEXT
    # =========================
    def render_text(self, text, x, y, color=None, font_scale=0.8, thickness=1):

        if color is None:
            color = COLORS['neon_blue']

        cv2.putText(self.frame, text, (x+2, y+2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness+2)

        cv2.putText(self.frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness)

    # =========================
    # EFFECT SAFE SCANLINE
    # =========================
    def render_hologram_scanlines(self):

        if self.frame is None:
            return

        h = self.frame.shape[0]

        # 🔥 FIX: safe loop + no None crash
        for y in range(0, h, 6):

            if y >= h:
                continue

            line = self.frame[y:y+1]
            if line is None or line.size == 0:
                continue

            self.frame[y:y+1] = cv2.addWeighted(
                line,
                0.75,
                np.zeros_like(line),
                0.25,
                0
            )

    def apply_vignette(self):

        rows, cols = self.frame.shape[:2]

        kernel_x = cv2.getGaussianKernel(cols, 250)
        kernel_y = cv2.getGaussianKernel(rows, 250)

        mask = kernel_y * kernel_x.T
        mask = mask / mask.max()

        for i in range(3):
            self.frame[:, :, i] = self.frame[:, :, i] * mask

    # =========================
    # INTERNAL UI
    # =========================
    def _draw_selection_box(self, cx, cy, w, h):

        color = COLORS['neon_blue']
        corner = 20

        ox = w // 2
        oy = h // 2

        points = [
            ((cx-ox, cy-oy), (cx-ox+corner, cy-oy)),
            ((cx-ox, cy-oy), (cx-ox, cy-oy+corner)),

            ((cx+ox, cy-oy), (cx+ox-corner, cy-oy)),
            ((cx+ox, cy-oy), (cx+ox, cy-oy+corner)),

            ((cx-ox, cy+oy), (cx-ox+corner, cy+oy)),
            ((cx-ox, cy+oy), (cx-ox, cy+oy-corner)),

            ((cx+ox, cy+oy), (cx+ox-corner, cy+oy)),
            ((cx+ox, cy+oy), (cx+ox, cy+oy-corner))
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
            COLORS['purple'],
            1
        )

    # =========================
    # OUTPUT
    # =========================
    def get_frame(self):

        self.render_hologram_scanlines()
        self.apply_vignette()

        return self.frame.copy()
