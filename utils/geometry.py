"""
Utility functions for ORVIXA.
Common helper functions for geometry, math, and data processing.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from utils.config import GESTURE_SMOOTHING


class Point2D:
    """Simple 2D point class."""
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point2D(self.x * scalar, self.y * scalar)
    
    def distance_to(self, other) -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))


class Vector2D:
    """2D vector operations."""
    
    @staticmethod
    def distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)
    
    @staticmethod
    def magnitude(v: np.ndarray) -> float:
        """Calculate magnitude of a vector."""
        return np.linalg.norm(v)
    
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        mag = np.linalg.norm(v)
        return v / mag if mag > 0 else v
    
    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in radians."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)


class SmoothingFilter:
    """Exponential moving average smoothing filter."""
    
    def __init__(self, alpha: float = GESTURE_SMOOTHING):
        """
        Initialize the smoothing filter.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = more smoothing.
        """
        self.alpha = alpha
        self.previous_value = None
    
    def smooth(self, value: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average smoothing.
        
        Args:
            value: Current value to smooth.
        
        Returns:
            Smoothed value.
        """
        if self.previous_value is None:
            self.previous_value = value.copy()
            return value
        
        smoothed = self.alpha * value + (1 - self.alpha) * self.previous_value
        self.previous_value = smoothed.copy()
        return smoothed
    
    def reset(self):
        """Reset the filter."""
        self.previous_value = None


def normalize_coordinates(point: Tuple[float, float], 
                         width: int, height: int) -> np.ndarray:
    """Convert pixel coordinates to normalized [0, 1] range."""
    x, y = point
    return np.array([x / width, y / height])


def denormalize_coordinates(norm_point: np.ndarray,
                           width: int, height: int) -> Tuple[int, int]:
    """Convert normalized coordinates back to pixel coordinates."""
    x, y = norm_point
    return (int(x * width), int(y * height))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values."""
    return a + (b - a) * t


def draw_circle_with_glow(image: np.ndarray, center: Tuple[int, int],
                         radius: int, color: Tuple[int, int, int],
                         thickness: int = 2, glow_strength: int = 3):
    """Draw a circle with glow effect."""
    # Draw outer glow
    for i in range(glow_strength, 0, -1):
        alpha = 0.3 * (1 - i / glow_strength)
        cv2.circle(image, center, radius + i, color, thickness=1)
    # Draw main circle
    cv2.circle(image, center, radius, color, thickness)


def draw_line_with_glow(image: np.ndarray, pt1: Tuple[int, int],
                       pt2: Tuple[int, int], color: Tuple[int, int, int],
                       thickness: int = 2, glow_strength: int = 2):
    """Draw a line with glow effect."""
    # Draw outer glow
    for i in range(glow_strength, 0, -1):
        cv2.line(image, pt1, pt2, color, thickness=(thickness + i))
    # Draw main line
    cv2.line(image, pt1, pt2, color, thickness)


def resize_with_aspect_ratio(image: np.ndarray, target_width: int,
                            target_height: int) -> np.ndarray:
    """
    Resize image maintaining aspect ratio.
    Pads with black if necessary.
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def create_transparent_overlay(width: int, height: int) -> np.ndarray:
    """Create a transparent overlay (RGBA)."""
    return np.zeros((height, width, 4), dtype=np.uint8)


def apply_alpha_blend(foreground: np.ndarray, background: np.ndarray,
                     alpha: float) -> np.ndarray:
    """
    Blend foreground and background images.
    
    Args:
        foreground: Foreground image (BGR).
        background: Background image (BGR).
        alpha: Alpha value (0-1).
    
    Returns:
        Blended image.
    """
    return cv2.addWeighted(foreground, alpha, background, 1 - alpha, 0)


def get_hand_center(landmarks: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Get the center point of a hand from its landmarks."""
    if not landmarks:
        return (0, 0)
    
    points = np.array(landmarks)
    center = np.mean(points, axis=0)
    return tuple(center)
