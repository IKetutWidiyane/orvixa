"""
Real-time camera capture module.
Handles webcam input with threading support.
"""

import cv2
import threading
import numpy as np
from collections import deque
from utils.logger import get_logger
from utils.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, FRAME_BUFFER_SIZE, ENABLE_THREADING


class CameraCapture:
    """Real-time camera capture handler with buffering."""
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (0 for default webcam).
        """
        self.logger = get_logger()
        self.camera_id = camera_id
        self.cap = None
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.is_running = False
        self.capture_thread = None
        self._lock = threading.Lock()
        self.frame_count = 0
        self.fps = 0
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera and set properties."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Get actual resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {CAMERA_FPS} FPS")
            
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            raise
    
    def start(self):
        """Start capture thread."""
        if self.is_running:
            return
        
        self.is_running = True
        
        if ENABLE_THREADING:
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            self.logger.info("Camera capture started (threaded mode)")
        else:
            self.logger.info("Camera capture started (sync mode)")
    
    def _capture_loop(self):
        """Continuous frame capture loop."""
        import time
        last_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            
            if ret:
                with self._lock:
                    self.frame_buffer.append(frame)
                    self.frame_count += 1
                
                # Update FPS
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    last_time = current_time
            else:
                self.logger.warning("Failed to capture frame")
                break
    
    def get_frame(self) -> np.ndarray:
        """
        Get the latest frame from buffer.
        
        Returns:
            Latest captured frame or None if no frame available.
        """
        if ENABLE_THREADING:
            with self._lock:
                if self.frame_buffer:
                    return self.frame_buffer[-1].copy()
                return None
        else:
            # Synchronous mode
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                return frame
            return None
    
    def get_fps(self) -> float:
        """Get current frames per second."""
        return self.fps
    
    def stop(self):
        """Stop camera capture."""
        self.is_running = False
        
        if ENABLE_THREADING and self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        self.logger.info("Camera capture stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
