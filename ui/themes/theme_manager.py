"""
Theme system for ORVIXA.
Manages colors, fonts, and UI styling.
"""

from typing import Tuple, Dict
from utils.config import COLORS


class Theme:
    """UI Theme definition."""
    
    def __init__(self, name: str):
        """Initialize theme."""
        self.name = name
        self.colors = {}
        self.fonts = {}
    
    def get_color(self, color_name: str) -> Tuple[int, int, int]:
        """Get color from theme."""
        return self.colors.get(color_name, (255, 255, 255))
    
    def set_color(self, color_name: str, color: Tuple[int, int, int]):
        """Set theme color."""
        self.colors[color_name] = color


class CyberpunkTheme(Theme):
    """Futuristic cyberpunk theme."""
    
    def __init__(self):
        """Initialize cyberpunk theme."""
        super().__init__("Cyberpunk")
        
        # Set up color palette
        self.colors = {
            'primary': COLORS['neon_blue'],        # Neon blue
            'secondary': COLORS['purple'],         # Purple
            'accent': (0, 255, 200),               # Cyan
            'warning': (0, 165, 255),              # Orange
            'danger': COLORS['red'],               # Red
            'success': (0, 255, 0),                # Green
            'background': COLORS['dark_bg'],       # Dark background
            'surface': COLORS['dark_gray'],        # Dark gray
            'text': (255, 255, 255),               # White text
            'text_secondary': (150, 150, 170),     # Dim text
            'border': COLORS['neon_blue'],         # Blue border
            'glow': COLORS['neon_blue'],           # Glow color
        }
    
    def get_gradient(self, width: int, height: int):
        """Get gradient background (placeholder)."""
        import numpy as np
        import cv2
        
        grad = np.zeros((height, width, 3), dtype=np.uint8)
        # Dark to darker gradient
        for y in range(height):
            ratio = y / height
            color = int(10 * ratio)
            grad[y, :] = [5 + color, 8 + color, 22 + color]
        
        return grad


class TerminalTheme(Theme):
    """Retro terminal theme."""
    
    def __init__(self):
        """Initialize terminal theme."""
        super().__init__("Terminal")
        
        self.colors = {
            'primary': (0, 255, 0),                # Green
            'secondary': (0, 128, 0),              # Dark green
            'accent': (0, 255, 255),               # Cyan
            'warning': (0, 255, 255),              # Cyan
            'danger': (0, 0, 255),                 # Red
            'success': (0, 255, 0),                # Green
            'background': (0, 0, 0),               # Black
            'surface': (10, 10, 10),               # Dark gray
            'text': (0, 255, 0),                   # Green text
            'text_secondary': (0, 128, 0),         # Dim green
            'border': (0, 255, 0),                 # Green border
            'glow': (0, 255, 0),                   # Green glow
        }


class ThemeManager:
    """Manages available themes."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize theme manager."""
        if not hasattr(self, 'themes'):
            self.themes = {}
            self.current_theme = None
            
            # Register themes
            self.register_theme(CyberpunkTheme())
            self.register_theme(TerminalTheme())
            
            # Set default theme
            self.set_theme('Cyberpunk')
    
    def register_theme(self, theme: Theme):
        """Register a new theme."""
        self.themes[theme.name] = theme
    
    def set_theme(self, theme_name: str) -> bool:
        """Set current theme."""
        if theme_name in self.themes:
            self.current_theme = self.themes[theme_name]
            return True
        return False
    
    def get_current_theme(self) -> Theme:
        """Get current active theme."""
        return self.current_theme
    
    def get_color(self, color_name: str) -> Tuple[int, int, int]:
        """Get color from current theme."""
        if self.current_theme:
            return self.current_theme.get_color(color_name)
        return (255, 255, 255)
    
    def list_themes(self):
        """Get list of available themes."""
        return list(self.themes.keys())


def get_theme_manager() -> ThemeManager:
    """Get global theme manager instance."""
    return ThemeManager()
