"""
Drawing system for air drawing in workspace.
Handles brush strokes, erasing, and color management.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from collections import deque
from utils.logger import get_logger
from utils.config import DRAWING_LINE_THICKNESS, DRAWING_COLOR, ERASER_SIZE
from utils.geometry import SmoothingFilter


class Brush:
    """Brush for drawing."""
    
    def __init__(self, color: Tuple[int, int, int] = DRAWING_COLOR,
                 thickness: int = DRAWING_LINE_THICKNESS):
        """
        Initialize brush.
        
        Args:
            color: BGR color tuple.
            thickness: Brush thickness in pixels.
        """
        self.color = color
        self.thickness = thickness
        self.is_eraser = False
    
    def set_color(self, color: Tuple[int, int, int]):
        """Set brush color."""
        self.color = color
    
    def set_thickness(self, thickness: int):
        """Set brush thickness."""
        self.thickness = max(1, min(thickness, 50))
    
    def toggle_eraser(self):
        """Toggle eraser mode."""
        self.is_eraser = not self.is_eraser


class DrawingCanvas:
    """Drawing canvas for air drawing."""
    
    def __init__(self, width: int, height: int):
        """
        Initialize drawing canvas.
        
        Args:
            width: Canvas width.
            height: Canvas height.
        """
        self.logger = get_logger()
        self.width = width
        self.height = height
        
        # Canvas with transparency (BGRA)
        self.canvas = np.zeros((height, width, 4), dtype=np.uint8)
        self.canvas[:, :, 3] = 255  # Full alpha
        
        # Drawing state
        self.brush = Brush()
        self.is_drawing = False
        self.stroke_points = deque(maxlen=100)  # Store recent points for smoothing
        self.point_filter = SmoothingFilter(alpha=0.7)
        
        self.logger.info(f"Drawing canvas created: {width}x{height}")
    
    def start_stroke(self, x: int, y: int):
        """Start a new drawing stroke."""
        self.is_drawing = True
        self.stroke_points.clear()
        self.point_filter.reset()
        self.stroke_points.append((int(x), int(y)))
    
    def add_stroke_point(self, x: int, y: int):
        """Add point to current stroke."""
        if not self.is_drawing:
            return
        
        point = np.array([x, y], dtype=np.float32)
        smoothed_point = self.point_filter.smooth(point)
        
        if len(self.stroke_points) > 0:
            last_point = self.stroke_points[-1]
            current_point = tuple(smoothed_point.astype(int))
            
            if self.brush.is_eraser:
                self._draw_eraser(last_point, current_point)
            else:
                self._draw_line(last_point, current_point)
        
        self.stroke_points.append(tuple(smoothed_point.astype(int)))
    
    def end_stroke(self):
        """End current drawing stroke."""
        self.is_drawing = False
        self.stroke_points.clear()
    
    def _draw_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """Draw line on canvas."""
        # Create temporary RGB canvas
        temp = cv2.cvtColor(self.canvas[:, :, :3], cv2.COLOR_BGR2RGB)
        
        # Draw anti-aliased line
        cv2.line(temp, pt1, pt2, self.brush.color, self.brush.thickness,
                cv2.LINE_AA)
        
        # Convert back
        self.canvas[:, :, :3] = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
    
    def _draw_eraser(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """Erase on canvas."""
        cv2.line(self.canvas, pt1, pt2, (0, 0, 0, 0), ERASER_SIZE, cv2.LINE_AA)
    
    def clear(self):
        """Clear entire canvas."""
        self.canvas.fill(0)
        self.canvas[:, :, 3] = 255  # Reset alpha
        self.logger.debug("Drawing canvas cleared")
    
    def undo(self):
        """Simple undo by clearing (not full undo stack)."""
        self.clear()
    
    def set_brush_color(self, color: Tuple[int, int, int]):
        """Set brush color."""
        self.brush.set_color(color)
    
    def set_brush_thickness(self, thickness: int):
        """Set brush thickness."""
        self.brush.set_thickness(thickness)
    
    def toggle_eraser(self):
        """Toggle eraser mode."""
        self.brush.toggle_eraser()
    
    def get_canvas(self) -> np.ndarray:
        """Get canvas image (BGR)."""
        return self.canvas[:, :, :3].copy()
    
    def get_canvas_with_alpha(self) -> np.ndarray:
        """Get canvas with transparency (BGRA)."""
        return self.canvas.copy()


class DrawingEngine:
    """High-level drawing engine."""
    
    def __init__(self, width: int, height: int):
        """Initialize drawing engine."""
        self.logger = get_logger()
        self.canvas = DrawingCanvas(width, height)
        self.is_active = False
        self.logger.info("Drawing engine initialized")
    
    def start_drawing(self, x: int, y: int):
        """Start drawing at position."""
        self.is_active = True
        self.canvas.start_stroke(x, y)
    
    def draw_at(self, x: int, y: int):
        """Draw at position."""
        if self.is_active:
            self.canvas.add_stroke_point(x, y)
    
    def stop_drawing(self):
        """Stop drawing."""
        self.is_active = False
        self.canvas.end_stroke()
    
    def clear(self):
        """Clear canvas."""
        self.canvas.clear()
    
    def set_brush_color(self, color: Tuple[int, int, int]):
        """Set brush color."""
        self.canvas.set_brush_color(color)
    
    def set_brush_size(self, size: int):
        """Set brush size."""
        self.canvas.set_brush_thickness(size)
    
    def toggle_eraser(self):
        """Toggle eraser."""
        self.canvas.toggle_eraser()
    
    def get_frame(self) -> np.ndarray:
        """Get drawing canvas for display."""
        return self.canvas.get_canvas()
