"""
Main ORVIXA Application.
Orchestrates all components into a cohesive spatial gesture interface.
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, List
import uuid

from utils.logger import get_logger
from utils.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, DEBUG_MODE,
    SHOW_LANDMARKS, GESTURE_SMOOTHING
)

from core.camera.capture import CameraCapture
from core.tracking.hand_tracker import HandTracker
from core.interaction.gesture_engine import GestureEngine, GestureType
from core.workspace.manager import VirtualWorkspace, WorkspaceObject
from core.rendering.renderer import Renderer
from core.effects.drawing import DrawingEngine
from core.effects.sound import SoundManager

from media.image.image_manager import ImageManager
from media.video.video_manager import VideoManager

from ui.themes import get_theme_manager
from ui.overlays.hud import HUDOverlay


class ORVIXA:
    """Main ORVIXA application class."""

    def __init__(self):
        """Initialize ORVIXA application."""
        self.logger = get_logger()
        self.logger.info("=" * 60)
        self.logger.info("ORVIXA - Spatial Gesture Workspace")
        self.logger.info("=" * 60)

        # Core components
        self.camera = CameraCapture()
        self.hand_tracker = HandTracker()
        self.gesture_engine = GestureEngine()
        self.workspace = VirtualWorkspace()
        self.renderer = Renderer(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.drawing_engine = DrawingEngine(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.sound_manager = SoundManager()

        # Media managers
        self.image_manager = ImageManager()
        self.video_manager = VideoManager()

        # UI
        self.theme_manager = get_theme_manager()
        self.hud = HUDOverlay(CAMERA_WIDTH, CAMERA_HEIGHT)

        # State
        self.is_running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Interaction state
        self.selected_hand = None
        self.drawing_mode = False
        self.current_tool = "cursor"
        self.hand_positions = {}
        self.active_drawing_hand = None

        # Performance
        self.frame_times = []

        self.logger.info("ORVIXA initialized successfully")

    def run(self):
        """Run the main application loop."""
        self.is_running = True
        self.logger.info("Starting main loop...")

        self.camera.start()
        time.sleep(0.5)

        try:
            while self.is_running:
                frame_start = time.time()

                frame = self.camera.get_frame()
                if frame is None:
                    continue

                output_frame = self._process_frame(frame)

                cv2.imshow('ORVIXA', output_frame)

                if not self._handle_input():
                    break

                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)

                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)

                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                self.frame_count += 1

                if self.frame_count % 300 == 0:
                    self.logger.info(
                        f"Running... FPS: {self.fps:.1f}, "
                        f"Objects: {len(self.workspace.objects)}"
                    )

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")

        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)

        finally:
            self.shutdown()

    # =========================
    # FRAME PROCESSING
    # =========================
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)

        self.renderer.set_background(frame)
        frame_height, frame_width = self.renderer.frame.shape[:2]
        self.hud.width = frame_width
        self.hud.height = frame_height
        self.drawing_engine.ensure_size(frame_width, frame_height)

        hands = self.hand_tracker.process(frame)
        gestures = self.gesture_engine.detect_gesture(hands)

        self._process_interactions(hands, gestures)
        self._process_drawing(hands, gestures)
        self._render_workspace()

        for hand in hands:
            self.renderer.render_hand(hand.landmarks)

            index_tip = hand.landmarks[8]

            cursor_x = int(index_tip[0] * frame_width)
            cursor_y = int(index_tip[1] * frame_height)

            self.renderer.render_cursor(cursor_x, cursor_y)

        if self.drawing_mode:
            drawing = self.drawing_engine.get_frame()
            mask = drawing > 0
            self.renderer.frame[mask] = drawing[mask]

        self.hud.update_fps(self.fps)
        self.hud.update_object_count(len(self.workspace.objects))
        self.hud.update_zoom(self.workspace.zoom)

        if gestures:
            self.hud.update_gesture(gestures[0].type.name)

        return self.hud.render(self.renderer.get_frame())

    # =========================
    # WORKSPACE RENDER
    # =========================
    def _render_workspace(self):
        self.renderer.render_workspace_background(self.workspace)

        for obj in self.workspace.get_sorted_objects():
            self.renderer.render_object(obj, self.workspace)

    # =========================
    # INTERACTIONS
    # =========================
    def _process_interactions(self, hands: List, gestures: List):
        if not gestures:
            return

        if self.drawing_mode:
            return

        for gesture in gestures:
            if gesture.type == GestureType.POINTING:
                self._handle_pointing(gesture)

            elif gesture.type == GestureType.PINCH:
                self._handle_pinch(gesture)

            elif gesture.type == GestureType.OPEN_PALM:
                self._handle_open_palm(gesture)

            elif gesture.type == GestureType.CLOSED_FIST:
                self._handle_closed_fist(gesture)

    def _handle_pointing(self, gesture):
        self.sound_manager.play_sound('click')

    def _handle_pinch(self, gesture):
        if gesture.metadata:
            distance = gesture.metadata.get('distance', 0)

            if distance < 0.03:
                self.workspace.zoom_in(1.05)
                self.sound_manager.play_sound('zoom')

    def _handle_open_palm(self, gesture):
        self.workspace.reset_view()
        self.logger.info("View reset")

    def _handle_closed_fist(self, gesture):
        if self.workspace.selected_object:
            self.workspace.remove_object(
                self.workspace.selected_object.object_id
            )
            self.logger.info("Object deleted")

    def _process_drawing(self, hands: List, gestures: List):
        if not self.drawing_mode:
            self._stop_active_stroke()
            return

        pointing_gesture = next(
            (gesture for gesture in gestures
             if gesture.type == GestureType.POINTING and gesture.metadata),
            None
        )

        if pointing_gesture is None:
            self._stop_active_stroke()
            return

        finger_pos = pointing_gesture.metadata.get('finger_pos')
        if finger_pos is None:
            self._stop_active_stroke()
            return

        frame_height, frame_width = self.renderer.frame.shape[:2]
        x = int(finger_pos[0] * frame_width)
        y = int(finger_pos[1] * frame_height)

        if self.active_drawing_hand != pointing_gesture.hand_id:
            self._stop_active_stroke()
            self.drawing_engine.start_drawing(x, y)
            self.active_drawing_hand = pointing_gesture.hand_id
        else:
            self.drawing_engine.draw_at(x, y)

    def _stop_active_stroke(self):
        if self.active_drawing_hand is not None:
            self.drawing_engine.stop_drawing()
            self.active_drawing_hand = None

    # =========================
    # INPUT
    # =========================
    def _handle_input(self) -> bool:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return False

        elif key == ord('c'):
            self.drawing_engine.clear()
            self.logger.info("Canvas cleared")

        elif key == ord('e'):
            self.drawing_engine.toggle_eraser()
            self._update_drawing_tool_label()

        elif key == ord('d'):
            self.drawing_mode = not self.drawing_mode
            self._stop_active_stroke()
            self._update_drawing_tool_label()
            self.logger.info(f"Drawing mode: {self.drawing_mode}")

        elif key == ord('r'):
            self.workspace.reset_view()

        elif key == ord('t'):
            self.current_tool = (
                "zoom" if self.current_tool == "cursor" else "cursor"
            )
            self.hud.update_tool(self.current_tool)

        elif key == ord('1'):
            self._load_sample_media()

        elif key in (ord('+'), ord('=')):
            self._change_brush_size(1)

        elif key in (ord('-'), ord('_')):
            self._change_brush_size(-1)

        elif key == ord('2'):
            self._set_brush_color((0, 217, 255), "neon blue")

        elif key == ord('3'):
            self._set_brush_color((0, 255, 217), "cyan")

        elif key == ord('4'):
            self._set_brush_color((138, 43, 226), "purple")

        elif key == ord('5'):
            self._set_brush_color((0, 255, 0), "green")

        return True

    def _load_sample_media(self):
        self.logger.info("Sample media loading not yet implemented")

    def _change_brush_size(self, delta: int):
        size = self.drawing_engine.get_brush_size() + delta
        self.drawing_engine.set_brush_size(size)
        self._update_drawing_tool_label()
        self.logger.info(f"Brush size: {self.drawing_engine.get_brush_size()}")

    def _set_brush_color(self, color, name: str):
        self.drawing_engine.set_brush_color(color)
        if self.drawing_engine.is_eraser_active():
            self.drawing_engine.toggle_eraser()
        self._update_drawing_tool_label()
        self.logger.info(f"Brush color: {name}")

    def _update_drawing_tool_label(self):
        if not self.drawing_mode:
            self.hud.update_tool(self.current_tool)
            return

        mode = "Eraser" if self.drawing_engine.is_eraser_active() else "Brush"
        size = self.drawing_engine.get_brush_size()
        self.hud.update_tool(f"Drawing: {mode} {size}px")

    # =========================
    # SHUTDOWN
    # =========================
    def shutdown(self):
        self.logger.info("Shutting down ORVIXA...")

        self.is_running = False
        self._stop_active_stroke()

        self.camera.stop()
        self.hand_tracker.close()
        self.video_manager.clear()

        cv2.destroyAllWindows()

        self.logger.info(f"Total frames processed: {self.frame_count}")
        self.logger.info("ORVIXA shutdown complete")


def main():
    app = ORVIXA()
    app.run()


if __name__ == "__main__":
    main()
