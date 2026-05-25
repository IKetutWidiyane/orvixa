"""
Hand tracking module using MediaPipe.
Detects and tracks hand landmarks in real-time.
"""

import numpy as np
import mediapipe as mp
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from utils.logger import get_logger
from utils.config import HAND_CONFIDENCE, MAX_HANDS, STATIC_IMAGE_MODE
from utils.geometry import SmoothingFilter


@dataclass
class Hand:
    """Data class for hand tracking information."""
    landmarks: np.ndarray  # 21 landmarks, each with (x, y, z)
    handedness: str  # 'Left' or 'Right'
    confidence: float  # Detection confidence
    is_left: bool


class HandTracker:
    """MediaPipe-based hand tracking."""
    
    def __init__(self):
        """Initialize hand tracker."""
        self.logger = get_logger()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=HAND_CONFIDENCE,
            min_tracking_confidence=HAND_CONFIDENCE
        )
        
        # Landmark smoothing
        self.smoothing_filters = {}
        
        self.logger.info(f"Hand tracker initialized (confidence: {HAND_CONFIDENCE})")
    
    def process(self, frame: np.ndarray) -> List[Hand]:
        """
        Process frame and detect hands.
        
        Args:
            frame: Input frame (BGR).
        
        Returns:
            List of detected Hand objects.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        hands = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Extract landmarks
                landmarks_array = np.array([
                    [lm.x, lm.y, lm.z] for lm in landmarks.landmark
                ])
                
                # Apply smoothing
                hand_id = handedness_info.classification[0].label
                if hand_id not in self.smoothing_filters:
                    self.smoothing_filters[hand_id] = SmoothingFilter()
                
                smoothed_landmarks = self.smoothing_filters[hand_id].smooth(landmarks_array)
                
                # Create Hand object
                hand = Hand(
                    landmarks=smoothed_landmarks,
                    handedness=handedness_info.classification[0].label,
                    confidence=handedness_info.classification[0].score,
                    is_left=handedness_info.classification[0].label == 'Left'
                )
                
                hands.append(hand)
        
        return hands
    
    def get_landmark(self, hand: Hand, index: int) -> np.ndarray:
        """Get specific landmark coordinates (normalized 0-1)."""
        return hand.landmarks[index]
    
    def get_finger_tip(self, hand: Hand, finger: str) -> np.ndarray:
        """
        Get specific finger tip position.
        
        Args:
            hand: Hand object.
            finger: 'thumb', 'index', 'middle', 'ring', 'pinky'.
        
        Returns:
            Normalized coordinates (x, y, z).
        """
        finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        if finger not in finger_tips:
            return None
        
        return hand.landmarks[finger_tips[finger]]
    
    def get_hand_center(self, hand: Hand) -> np.ndarray:
        """Get center of hand (average of all landmarks)."""
        return np.mean(hand.landmarks, axis=0)
    
    def get_hand_bbox(self, hand: Hand, frame_width: int,
                     frame_height: int) -> Tuple[int, int, int, int]:
        """
        Get bounding box of hand.
        
        Returns:
            (x_min, y_min, x_max, y_max) in pixel coordinates.
        """
        landmarks = hand.landmarks[:, :2]  # Only x, y
        
        x_min = int(np.min(landmarks[:, 0]) * frame_width)
        x_max = int(np.max(landmarks[:, 0]) * frame_width)
        y_min = int(np.min(landmarks[:, 1]) * frame_height)
        y_max = int(np.max(landmarks[:, 1]) * frame_height)
        
        return (x_min, y_min, x_max, y_max)
    
    def reset(self):
        """Reset smoothing filters."""
        self.smoothing_filters.clear()
    
    def close(self):
        """Clean up resources."""
        self.hands.close()
        self.logger.info("Hand tracker closed")


# Import here to avoid circular dependency
import cv2
