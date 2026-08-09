"""
Main ORVIXA Application.
Orchestrates all components into a cohesive spatial gesture interface.
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, List
import uuid

from utils.logger import get_logger
from utils.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, DEBUG_MODE,
    SHOW_LANDMARKS, GESTURE_SMOOTHING,
    IMAGES_PATH, VIDEOS_PATH
)

from core.camera.capture import CameraCapture
from core.tracking.hand_tracker import HandTracker
from core.interaction.gesture_engine import GestureEngine, GestureType
from core.workspace.manager import VirtualWorkspace, WorkspaceObject
from core.rendering.renderer import Renderer
from core.effects.drawing import DrawingEngine
from core.effects.sound import SoundManager
from core.shapes.shape_manager import ShapeManager, ShapeType

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

        # Shape system
        self.shape_manager = ShapeManager()
        self.current_shape = ShapeType.RECTANGLE

        # Interaction state
        self.selected_hand = None
        self.drawing_mode = False
        self.current_tool = "cursor"
        self.hand_positions = {}
        self.active_drawing_hand = None
        self.drag_state = {}
        self.hud.update_shape(self.current_shape.value)

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

        self.video_manager.update_all()
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
        # Handle completed shape drawing when PEACE gesture stops
        if not gestures:
            self.shape_manager.process_two_fingers(None, None)
            self.drag_state.clear()
            return

        if self.drawing_mode:
            self.shape_manager.process_two_fingers(None, None)
            self.drag_state.clear()
            return

        # Check if PEACE gesture is present, if not complete any shape drawing
        has_peace = any(g.type == GestureType.PEACE for g in gestures)
        if not has_peace:
            new_object = self.shape_manager.process_two_fingers(None, None)
            if new_object:
                self.workspace.add_object(new_object)
                self.logger.info(f"Shape created: {new_object.media_type}")

        for gesture in gestures:
            if gesture.type == GestureType.POINTING:
                self._handle_pointing(gesture)

            elif gesture.type == GestureType.PINCH:
                self._handle_pinch(gesture)

            elif gesture.type == GestureType.PEACE:
                self._handle_shape_drawing(gesture)

            elif gesture.type == GestureType.GRABBING:
                self._handle_drag(gesture)

            elif gesture.type == GestureType.THREE_FINGERS:
                self._handle_rotate(gesture)

            elif gesture.type == GestureType.OPEN_PALM:
                self._handle_open_palm(gesture)

            elif gesture.type == GestureType.CLOSED_FIST:
                self._handle_closed_fist(gesture)

    def _handle_pointing(self, gesture):
        pos = self._gesture_position(gesture)
        if pos is None:
            return

        world_x, world_y = self._screen_to_world_normalized(pos)
        previous = self.workspace.selected_object
        selected = self.workspace.select_object_at(world_x, world_y)

        if selected and selected is not previous:
            self.sound_manager.play_sound('click')

    def _handle_pinch(self, gesture):
        if gesture.metadata:
            distance = gesture.metadata.get('distance', 0)

            if distance < 0.03:
                if self.workspace.selected_object:
                    self.workspace.selected_object.resize(1.03)
                else:
                    self.workspace.zoom_in(1.05)
                self.sound_manager.play_sound('zoom')

    def _handle_drag(self, gesture):
        pos = self._gesture_position(gesture)
        if pos is None:
            return

        world_x, world_y = self._screen_to_world_normalized(pos)
        selected = self.workspace.selected_object

        if selected is None:
            selected = self.workspace.select_object_at(world_x, world_y)

        if selected is None or selected.locked:
            return

        state = self.drag_state.get(gesture.hand_id)
        if state is None:
            self.drag_state[gesture.hand_id] = (world_x, world_y)
            return

        last_x, last_y = state
        selected.move(world_x - last_x, world_y - last_y)
        self.drag_state[gesture.hand_id] = (world_x, world_y)

    def _handle_shape_drawing(self, gesture):
        """Create shapes by drawing with two fingers (PEACE gesture)."""
        if not gesture.metadata:
            return

        finger_pos = gesture.metadata.get('finger_pos')
        if finger_pos is None:
            self.shape_manager.cancel_drawing()
            return

        # Estimate middle finger position from finger distance
        # Middle finger is positioned to one side of the index finger
        middle_pos = None
        if 'finger_distance' in gesture.metadata:
            distance = gesture.metadata.get('finger_distance', 0.05)
            # Offset middle finger perpendicular to index finger
            # (approx. 60% of the index-to-middle distance to the right-down side)
            offset = distance * 0.6
            middle_pos = (
                float(finger_pos[0]) + offset,
                float(finger_pos[1]) + offset * 0.5
            )

        new_object = self.shape_manager.process_two_fingers(
            hand_id=gesture.hand_id,
            index_pos=(float(finger_pos[0]), float(finger_pos[1])),
            middle_pos=middle_pos,
            default_shape=self.current_shape
        )

        if new_object:
            self.workspace.add_object(new_object)
            self.sound_manager.play_sound('hologram')
            self.logger.info(f"Shape created: {new_object.media_type}")

    def _handle_rotate(self, gesture):
        selected = self.workspace.selected_object
        if selected:
            selected.rotate(3.0)

    def _handle_open_palm(self, gesture):
        self.workspace.reset_view()
        self.logger.info("View reset")

    def _handle_closed_fist(self, gesture):
        if self.workspace.selected_object:
            object_id = self.workspace.selected_object.object_id
            self.workspace.remove_object(
                object_id
            )
            self.image_manager.remove_image(object_id)
            self.video_manager.remove_video(object_id)
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

        # Shape selection menu (F1-F6 or digit keys 6-0)
        elif key == ord('6'):
            self._change_shape(ShapeType.RECTANGLE)

        elif key == ord('7'):
            self._change_shape(ShapeType.CIRCLE)

        elif key == ord('8'):
            self._change_shape(ShapeType.TRIANGLE)

        elif key == ord('9'):
            self._change_shape(ShapeType.DIAMOND)

        elif key == ord('0'):
            self._change_shape(ShapeType.STAR)

        elif key == ord('h'):
            self._change_shape(ShapeType.HEXAGON)

        elif key == ord('s'):
            self._cycle_shape()

        return True

    def _change_shape(self, shape_type: ShapeType):
        """Set and display the active shape type."""
        self.current_shape = shape_type
        self.hud.update_shape(shape_type.value)
        self.logger.info(f"Active shape: {shape_type.value}")

    def _cycle_shape(self):
        """Cycle through all available shape types."""
        shapes = list(ShapeType)
        current_index = shapes.index(self.current_shape)
        next_index = (current_index + 1) % len(shapes)
        self._change_shape(shapes[next_index])

    def _load_sample_media(self):
        loaded = 0

        for image_path in self._iter_media_files(IMAGES_PATH, {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}):
            loaded += self._add_image_media(str(image_path), loaded)

        for video_path in self._iter_media_files(VIDEOS_PATH, {'.mp4', '.avi', '.mov', '.mkv', '.webm'}):
            loaded += self._add_video_media(str(video_path), loaded)

        if loaded == 0:
            self._add_demo_media()
            loaded = 1

        self.sound_manager.play_sound('hologram')
        self.logger.info(f"Loaded {loaded} media object(s)")

    def _iter_media_files(self, folder: str, extensions: set):
        path = Path(folder)
        if not path.exists():
            return []
        return sorted(
            file for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in extensions
        )

    def _add_image_media(self, image_path: str, index: int) -> int:
        object_id = f"image_{uuid.uuid4().hex[:8]}"
        obj = self.image_manager.load_image(
            object_id,
            image_path,
            -250 + index * 120,
            -120 + index * 70
        )
        if obj and self.workspace.add_object(obj):
            return 1
        return 0

    def _add_video_media(self, video_path: str, index: int) -> int:
        object_id = f"video_{uuid.uuid4().hex[:8]}"
        obj = self.video_manager.load_video(
            object_id,
            video_path,
            -250 + index * 120,
            -120 + index * 70
        )
        if obj and self.workspace.add_object(obj):
            obj.play()
            return 1
        return 0

    def _add_demo_media(self):
        object_id = f"demo_{uuid.uuid4().hex[:8]}"
        obj = WorkspaceObject(object_id, 0, 0, 420, 260)
        obj.media_type = "demo"
        obj.get_display_frame = self._create_demo_media_frame
        self.workspace.add_object(obj)

    def _create_demo_media_frame(self):
        frame = np.zeros((260, 420, 3), dtype=np.uint8)
        frame[:] = (16, 18, 34)
        cv2.rectangle(frame, (0, 0), (419, 259), (0, 217, 255), 3)
        cv2.line(frame, (30, 210), (390, 50), (0, 255, 217), 3)
        cv2.circle(frame, (105, 95), 42, (138, 43, 226), -1)
        cv2.putText(frame, "ORVIXA MEDIA", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 217, 255), 2)
        cv2.putText(frame, "Drop files in assets/images or assets/videos", (34, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame

    def _gesture_position(self, gesture):
        if not gesture.metadata:
            return None
        if 'finger_pos' in gesture.metadata:
            return gesture.metadata['finger_pos']
        if 'palm_pos' in gesture.metadata:
            return gesture.metadata['palm_pos']
        return None

    def _screen_to_world_normalized(self, pos):
        frame_height, frame_width = self.renderer.frame.shape[:2]
        screen_x = int(pos[0] * frame_width)
        screen_y = int(pos[1] * frame_height)
        return self.workspace.screen_to_world(
            screen_x,
            screen_y,
            frame_width,
            frame_height
        )

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
