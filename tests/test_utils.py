"""
Unit tests for ORVIXA components.
Run with: pytest tests/
"""

import pytest
import numpy as np
from utils.geometry import Point2D, Vector2D, SmoothingFilter, clamp, lerp
from utils.config import CAMERA_WIDTH, CAMERA_HEIGHT


class TestPoint2D:
    """Tests for Point2D class."""
    
    def test_point_creation(self):
        p = Point2D(10, 20)
        assert p.x == 10
        assert p.y == 20
    
    def test_point_distance(self):
        p1 = Point2D(0, 0)
        p2 = Point2D(3, 4)
        assert p1.distance_to(p2) == 5.0
    
    def test_point_to_tuple(self):
        p = Point2D(1.5, 2.5)
        assert p.to_tuple() == (1.5, 2.5)


class TestVector2D:
    """Tests for Vector2D utilities."""
    
    def test_distance(self):
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])
        assert Vector2D.distance(p1, p2) == 5.0
    
    def test_magnitude(self):
        v = np.array([3, 4])
        assert Vector2D.magnitude(v) == 5.0
    
    def test_normalize(self):
        v = np.array([3, 4])
        normalized = Vector2D.normalize(v)
        assert abs(Vector2D.magnitude(normalized) - 1.0) < 0.01


class TestSmoothingFilter:
    """Tests for smoothing filter."""
    
    def test_smoothing_reduces_variance(self):
        f = SmoothingFilter(alpha=0.5)
        
        values = [10, 50, 20, 60, 30]
        smoothed_values = []
        
        for v in values:
            smoothed = f.smooth(np.array([v]))
            smoothed_values.append(smoothed[0])
        
        # Smoothed values should be less extreme than original
        assert max(smoothed_values) < 60
        assert min(smoothed_values) > 10
    
    def test_reset(self):
        f = SmoothingFilter()
        f.smooth(np.array([10]))
        f.reset()
        
        # After reset, should return the value as-is
        result = f.smooth(np.array([50]))
        assert result[0] == 50


class TestMathUtilities:
    """Tests for math utilities."""
    
    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10
    
    def test_lerp(self):
        assert lerp(0, 10, 0.0) == 0
        assert lerp(0, 10, 0.5) == 5
        assert lerp(0, 10, 1.0) == 10


class TestConfiguration:
    """Tests for configuration."""
    
    def test_camera_settings(self):
        assert CAMERA_WIDTH > 0
        assert CAMERA_HEIGHT > 0
    
    def test_workspace_dimensions(self):
        from utils.config import WORKSPACE_WIDTH, WORKSPACE_HEIGHT
        assert WORKSPACE_WIDTH > CAMERA_WIDTH
        assert WORKSPACE_HEIGHT > CAMERA_HEIGHT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
