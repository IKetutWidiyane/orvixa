"""
Integration tests for ORVIXA components.
"""

import pytest
import numpy as np
import time
from core.workspace.manager import VirtualWorkspace, WorkspaceObject
from core.interaction.gesture_engine import GestureEngine, GestureType
from ui.themes import get_theme_manager


class TestWorkspaceIntegration:
    """Integration tests for workspace system."""
    
    def test_object_lifecycle(self):
        """Test adding, selecting, and removing objects."""
        ws = VirtualWorkspace()
        
        # Add object
        obj = WorkspaceObject("test_obj", 100, 100, 200, 150)
        assert ws.add_object(obj)
        assert len(ws.objects) == 1
        
        # Select object
        selected = ws.select_object_at(100, 100)
        assert selected is not None
        assert selected.object_id == "test_obj"
        assert selected.selected
        
        # Remove object
        assert ws.remove_object("test_obj")
        assert len(ws.objects) == 0
    
    def test_camera_transformations(self):
        """Test world-to-screen coordinate transformations."""
        ws = VirtualWorkspace()
        
        # Test identity transformation
        world_x, world_y = 100, 100
        screen_x, screen_y = ws.world_to_screen(world_x, world_y, 1920, 1080)
        
        # Should be around center
        assert 900 < screen_x < 1000
        assert 500 < screen_y < 600
        
        # Reverse transformation
        back_x, back_y = ws.screen_to_world(screen_x, screen_y, 1920, 1080)
        
        # Should be close to original
        assert abs(back_x - world_x) < 1
        assert abs(back_y - world_y) < 1
    
    def test_zoom_functionality(self):
        """Test zoom in/out."""
        ws = VirtualWorkspace()
        
        initial_zoom = ws.zoom
        ws.zoom_in()
        assert ws.zoom > initial_zoom
        
        ws.zoom_out()
        assert ws.zoom < ws.zoom
        
        ws.reset_view()
        assert ws.zoom == 1.0


class TestGestureEngineIntegration:
    """Integration tests for gesture engine."""
    
    def test_gesture_history(self):
        """Test gesture history tracking."""
        engine = GestureEngine()
        
        # Create mock gestures
        from core.interaction.gesture_engine import Gesture
        
        gesture1 = Gesture(GestureType.PINCH, 0.9, 0, time.time())
        gesture2 = Gesture(GestureType.POINTING, 0.85, 0, time.time())
        
        engine.gestures_history.append(gesture1)
        engine.gestures_history.append(gesture2)
        
        # Get last gesture
        last = engine.get_last_gesture()
        assert last.type == GestureType.POINTING
        
        # Get gesture history
        pinch_history = engine.get_gesture_history(GestureType.PINCH)
        assert len(pinch_history) == 1


class TestThemeIntegration:
    """Integration tests for theme system."""
    
    def test_theme_switching(self):
        """Test switching between themes."""
        tm = get_theme_manager()
        
        # Switch to terminal theme
        assert tm.set_theme('Terminal')
        theme = tm.get_current_theme()
        assert theme.name == 'Terminal'
        
        # Switch back to cyberpunk
        assert tm.set_theme('Cyberpunk')
        theme = tm.get_current_theme()
        assert theme.name == 'Cyberpunk'
    
    def test_theme_colors(self):
        """Test retrieving theme colors."""
        tm = get_theme_manager()
        tm.set_theme('Cyberpunk')
        
        # Get primary color
        primary = tm.get_color('primary')
        assert isinstance(primary, tuple)
        assert len(primary) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
