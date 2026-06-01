"""
Image media handling.
Loads and manages image objects in workspace.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from utils.logger import get_logger
from core.workspace.manager import WorkspaceObject


class ImageObject(WorkspaceObject):
    """Image media object."""
    
    def __init__(self, object_id: str, image_path: str, x: float, y: float,
                 width: float = 400, height: float = 300):
        """
        Create image object.
        
        Args:
            object_id: Unique object identifier.
            image_path: Path to image file.
            x, y: Initial position.
            width, height: Display dimensions.
        """
        super().__init__(object_id, x, y, width, height)
        self.media_type = "image"
        self.source_path = image_path
        
        self.image_path = image_path
        self.image = None
        self.original_image = None
        self.logger = get_logger()
        
        self._load_image()
    
    def _load_image(self):
        """Load image from file."""
        try:
            path = Path(self.image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {self.image_path}")
            
            self.original_image = cv2.imread(str(path))
            if self.original_image is None:
                raise ValueError(f"Failed to load image: {self.image_path}")
            
            self.image = self.original_image.copy()
            self.logger.info(f"Image loaded: {path.name}")
            
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            self.image = None
    
    def get_display_image(self) -> Optional[np.ndarray]:
        """Get image resized to display dimensions."""
        if self.image is None:
            return None
        
        # Resize to display dimensions
        display = cv2.resize(self.image, 
                            (int(self.width * self.scale),
                             int(self.height * self.scale)))
        
        return display

    def get_display_frame(self) -> Optional[np.ndarray]:
        """Renderer-compatible frame accessor."""
        return self.get_display_image()
    
    def set_opacity(self, opacity: float):
        """Set image opacity (0-1)."""
        self.opacity = np.clip(opacity, 0.0, 1.0)


class ImageManager:
    """Manager for image objects."""
    
    def __init__(self):
        """Initialize image manager."""
        self.logger = get_logger()
        self.images = {}
        self.logger.info("Image manager initialized")
    
    def load_image(self, object_id: str, image_path: str,
                  x: float, y: float) -> Optional[ImageObject]:
        """Load and create image object."""
        try:
            img_obj = ImageObject(object_id, image_path, x, y)
            if img_obj.image is not None:
                self.images[object_id] = img_obj
                return img_obj
            return None
        except Exception as e:
            self.logger.error(f"Error creating image object: {e}")
            return None
    
    def get_image(self, object_id: str) -> Optional[ImageObject]:
        """Get image object by ID."""
        return self.images.get(object_id)
    
    def remove_image(self, object_id: str) -> bool:
        """Remove image object."""
        if object_id in self.images:
            del self.images[object_id]
            return True
        return False
    
    def clear(self):
        """Clear all images."""
        self.images.clear()
