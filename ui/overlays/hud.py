"""
UI Overlays for HUD and status display.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from utils.config import COLORS
from ui.themes import get_theme_manager


class HUDOverlay:
    """Head-up display overlay."""
    
    def __init__(self, width: int, height: int):
        """
        Initialize HUD overlay.
        
        Args:
            width: Screen width.
            height: Screen height.
        """
        self.width = width
        self.height = height
        self.theme = get_theme_manager()
        
        self.fps = 0
        self.gesture_status = "None"
        self.active_tool = "Cursor"
        self.object_count = 0
        self.zoom_level = 1.0
        self.active_shape = "rectangle"
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render HUD onto frame."""
        overlay = frame.copy()
        
        # Render FPS
        self._render_fps(overlay)
        
        # Render status panel
        self._render_status_panel(overlay)
        
        # Render bottom info
        self._render_bottom_info(overlay)
        
        return overlay
    
    def _render_fps(self, frame: np.ndarray):
        """Render FPS counter."""
        text = f"FPS: {self.fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = COLORS['neon_blue']
        
        cv2.putText(frame, text, (20, 40), font, 1.0, color, 2)
    
    def _render_status_panel(self, frame: np.ndarray):
        """Render status information panel."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = COLORS['neon_blue']
        secondary_color = COLORS['purple']
        
        y_pos = 80
        line_height = 30
        
        # Draw panel background
        panel_x1, panel_y1 = 20, y_pos - 20
        panel_x2, panel_y2 = 350, y_pos + line_height * 4
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2),
                     COLORS['dark_gray'], -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2),
                     color, 2)
        
        # Text
        cv2.putText(frame, f"Gesture: {self.gesture_status}", (40, y_pos),
                   font, 0.6, color, 1)
        cv2.putText(frame, f"Tool: {self.active_tool}", (40, y_pos + line_height),
                   font, 0.6, color, 1)
        cv2.putText(frame, f"Objects: {self.object_count}", (40, y_pos + line_height * 2),
                   font, 0.6, color, 1)
        cv2.putText(frame, f"Zoom: {self.zoom_level:.2f}x", (40, y_pos + line_height * 3),
                   font, 0.6, color, 1)
        cv2.putText(frame, f"Shape: {self.active_shape}", (40, y_pos + line_height * 4),
                   font, 0.6, secondary_color, 1)
    
    def _render_bottom_info(self, frame: np.ndarray):
        """Render bottom information bar."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = COLORS['neon_blue']
        
        y_pos = self.height - 30
        
        # Instructions
        instructions = [
            "Q: Exit | D: Drawing | E: Eraser | C: Clear | +/-: Brush | 2-5: Color",
            "Shape: 6:Rect 7:Circle 8:Triangle 9:Diamond 0:Star S:Cycle",
            "Gesture: PEACE (2 fingers) to draw shape | Draw by moving fingers apart"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (20, y_pos - i * 30),
                       font, 0.5, color, 1)
    
    def update_fps(self, fps: float):
        """Update FPS display."""
        self.fps = fps
    
    def update_gesture(self, gesture_name: str):
        """Update gesture status."""
        self.gesture_status = gesture_name
    
    def update_tool(self, tool_name: str):
        """Update active tool."""
        self.active_tool = tool_name
    
    def update_object_count(self, count: int):
        """Update object count."""
        self.object_count = count
    
    def update_zoom(self, zoom: float):
        """Update zoom level."""
        self.zoom_level = zoom

    def update_shape(self, shape_name: str):
        """Update active shape display."""
        self.active_shape = shape_name


class CornerIndicator:
    """Corner status indicator."""
    
    def __init__(self, width: int, height: int):
        """Initialize corner indicator."""
        self.width = width
        self.height = height
        self.indicators = []
    
    def add_indicator(self, corner: str, text: str, color: Tuple[int, int, int]):
        """
        Add indicator to corner.
        
        Args:
            corner: 'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left), 'br' (bottom-right).
            text: Text to display.
            color: RGB color.
        """
        self.indicators.append((corner, text, color))
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render indicators."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        positions = {
            'tl': (20, 30),
            'tr': (self.width - 200, 30),
            'bl': (20, self.height - 20),
            'br': (self.width - 200, self.height - 20),
        }
        
        for corner, text, color in self.indicators:
            if corner in positions:
                x, y = positions[corner]
                cv2.putText(frame, text, (x, y), font, 0.6, color, 1)
        
        return frame
