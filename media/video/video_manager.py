"""
Video media handling.
Loads and plays video objects in workspace.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from utils.logger import get_logger
from core.workspace.manager import WorkspaceObject


class VideoObject(WorkspaceObject):
    """Video media object."""
    
    def __init__(self, object_id: str, video_path: str, x: float, y: float,
                 width: float = 640, height: float = 480):
        """
        Create video object.
        
        Args:
            object_id: Unique object identifier.
            video_path: Path to video file.
            x, y: Initial position.
            width, height: Display dimensions.
        """
        super().__init__(object_id, x, y, width, height)
        self.media_type = "video"
        self.source_path = video_path
        
        self.video_path = video_path
        self.cap = None
        self.current_frame = None
        self.is_playing = False
        self.frame_index = 0
        self.total_frames = 0
        self.fps = 30
        self.logger = get_logger()
        
        self._load_video()
    
    def _load_video(self):
        """Load video file."""
        try:
            path = Path(self.video_path)
            if not path.exists():
                raise FileNotFoundError(f"Video not found: {self.video_path}")
            
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video: {self.video_path}")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Read first frame
            self._read_frame()
            
            self.logger.info(f"Video loaded: {path.name} ({self.total_frames} frames, {self.fps} FPS)")
            
        except Exception as e:
            self.logger.error(f"Error loading video: {e}")
            self.cap = None
    
    def _read_frame(self) -> bool:
        """Read next frame from video."""
        if self.cap is None:
            return False
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            return True
        return False
    
    def play(self):
        """Start playback."""
        self.is_playing = True
    
    def pause(self):
        """Pause playback."""
        self.is_playing = False
    
    def stop(self):
        """Stop playback and reset to beginning."""
        self.is_playing = False
        self.frame_index = 0
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._read_frame()
    
    def seek(self, frame_number: int):
        """Seek to specific frame."""
        if self.cap is None:
            return False
        
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return self._read_frame()
    
    def get_display_frame(self) -> Optional[np.ndarray]:
        """Get current frame for display."""
        if self.current_frame is None:
            return None
        
        # Resize to display dimensions
        display = cv2.resize(self.current_frame,
                            (int(self.width * self.scale),
                             int(self.height * self.scale)))
        
        return display
    
    def update(self):
        """Update video playback (should be called each frame)."""
        if self.is_playing and self.cap:
            if not self._read_frame():
                # Loop video
                self.seek(0)
    
    def close(self):
        """Close video file."""
        if self.cap:
            self.cap.release()
            self.cap = None


class VideoManager:
    """Manager for video objects."""
    
    def __init__(self):
        """Initialize video manager."""
        self.logger = get_logger()
        self.videos = {}
        self.logger.info("Video manager initialized")
    
    def load_video(self, object_id: str, video_path: str,
                  x: float, y: float) -> Optional[VideoObject]:
        """Load and create video object."""
        try:
            video_obj = VideoObject(object_id, video_path, x, y)
            if video_obj.cap is not None:
                self.videos[object_id] = video_obj
                return video_obj
            return None
        except Exception as e:
            self.logger.error(f"Error creating video object: {e}")
            return None
    
    def get_video(self, object_id: str) -> Optional[VideoObject]:
        """Get video object by ID."""
        return self.videos.get(object_id)
    
    def remove_video(self, object_id: str) -> bool:
        """Remove video object."""
        if object_id in self.videos:
            self.videos[object_id].close()
            del self.videos[object_id]
            return True
        return False
    
    def update_all(self):
        """Update all playing videos."""
        for video in self.videos.values():
            if video.is_playing:
                video.update()
    
    def clear(self):
        """Clear all videos."""
        for video in self.videos.values():
            video.close()
        self.videos.clear()
