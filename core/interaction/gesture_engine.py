"""
Gesture recognition and interaction engine.
Detects and processes hand gestures.
"""

import numpy as np
import time
from typing import Optional, Dict, List
from enum import Enum
from dataclasses import dataclass
from utils.logger import get_logger
from utils.config import PINCH_THRESHOLD, PINCH_COOLDOWN
from core.tracking.hand_tracker import Hand


class GestureType(Enum):
    """Available gesture types."""
    NONE = 0
    POINTING = 1  # One finger up (cursor mode)
    GRABBING = 2  # Multiple fingers bent (drag mode)
    PINCH = 3  # Thumb and index finger close
    OPEN_PALM = 4  # All fingers open
    CLOSED_FIST = 5  # All fingers closed
    PEACE = 6  # Two fingers up
    THREE_FINGERS = 7  # Three fingers up
    THUMBS_UP = 8


@dataclass
class Gesture:
    """Gesture detection result."""
    type: GestureType
    confidence: float
    hand_id: int  # 0 for left, 1 for right
    timestamp: float
    metadata: Dict = None  # Additional gesture data


class GestureEngine:
    """Hand gesture recognition and processing."""
    
    def __init__(self):
        """Initialize gesture engine."""
        self.logger = get_logger()
        self.last_pinch_time = {}  # Track pinch cooldown per hand
        self.gestures_history = []
        self.max_history = 100
        self.logger.info("Gesture engine initialized")
    
    def detect_gesture(self, hands: List[Hand]) -> List[Gesture]:
        """
        Detect gestures from hand tracking data.
        
        Args:
            hands: List of Hand objects from tracker.
        
        Returns:
            List of detected Gesture objects.
        """
        gestures = []
        
        for hand_idx, hand in enumerate(hands):
            gesture = self._analyze_hand(hand, hand_idx)
            if gesture:
                gestures.append(gesture)
                self.gestures_history.append(gesture)
                
                # Keep history size manageable
                if len(self.gestures_history) > self.max_history:
                    self.gestures_history.pop(0)
        
        return gestures
    
    def _analyze_hand(self, hand: Hand, hand_idx: int) -> Optional[Gesture]:
        """Analyze a single hand and detect gesture."""
        landmarks = hand.landmarks[:, :2]  # Use x, y only
        
        # Detect individual finger states
        fingers_up = self._detect_fingers_up(landmarks)
        thumb_index_distance = self._get_finger_distance(landmarks, 4, 8)
        palm_openness = self._calculate_palm_openness(landmarks)
        
        # Determine gesture type based on finger configuration
        if thumb_index_distance < PINCH_THRESHOLD:
            gesture_type = GestureType.PINCH
            metadata = {'distance': thumb_index_distance}
        elif np.sum(fingers_up) == 1 and fingers_up[1]:  # Only index finger
            gesture_type = GestureType.POINTING
            metadata = {'finger_pos': landmarks[8]}
        elif np.sum(fingers_up) == 2 and fingers_up[1] and fingers_up[2]:  # Peace sign
            gesture_type = GestureType.PEACE
            metadata = {'finger_distance': self._get_finger_distance(landmarks, 8, 12)}
        elif np.sum(fingers_up) == 3:
            gesture_type = GestureType.THREE_FINGERS
            metadata = {'fingers_up': fingers_up.tolist()}
        elif np.sum(fingers_up) == 0:  # Closed fist
            gesture_type = GestureType.CLOSED_FIST
            metadata = {}
        elif np.all(fingers_up):  # Open palm
            gesture_type = GestureType.OPEN_PALM
            metadata = {'palm_openness': palm_openness}
        else:
            gesture_type = GestureType.GRABBING
            metadata = {'fingers_up': fingers_up.tolist()}
        
        # Calculate confidence (simplified)
        confidence = hand.confidence
        
        # Create gesture object
        gesture = Gesture(
            type=gesture_type,
            confidence=confidence,
            hand_id=hand_idx,
            timestamp=time.time(),
            metadata=metadata
        )
        
        return gesture
    
    def _detect_fingers_up(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Detect which fingers are up.
        
        Returns:
            Boolean array [thumb, index, middle, ring, pinky].
        """
        # Finger tip indices: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        # Finger PIP indices: 3, 6, 10, 14, 18
        
        fingers_up = np.zeros(5, dtype=bool)
        
        # Thumb
        fingers_up[0] = landmarks[4][0] < landmarks[3][0]  # x coordinate
        
        # Other fingers
        for finger_idx, (tip, pip) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)], 1):
            fingers_up[finger_idx] = landmarks[tip][1] < landmarks[pip][1]  # y coordinate
        
        return fingers_up
    
    def _get_finger_distance(self, landmarks: np.ndarray,
                            idx1: int, idx2: int) -> float:
        """Calculate distance between two landmarks."""
        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        return np.linalg.norm(p2 - p1)
    
    def _calculate_palm_openness(self, landmarks: np.ndarray) -> float:
        """Calculate how open the palm is (0-1)."""
        # Calculate distance between finger tips
        finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
        palm_center = landmarks[9]  # Center of palm
        
        distances = []
        for tip_idx in finger_tips:
            dist = np.linalg.norm(landmarks[tip_idx] - palm_center)
            distances.append(dist)
        
        # Openness is the average distance from palm center to finger tips
        openness = np.mean(distances)
        return np.clip(openness, 0, 1)
    
    def get_last_gesture(self, hand_id: Optional[int] = None) -> Optional[Gesture]:
        """Get the most recent gesture."""
        if not self.gestures_history:
            return None
        
        if hand_id is not None:
            # Get last gesture for specific hand
            for gesture in reversed(self.gestures_history):
                if gesture.hand_id == hand_id:
                    return gesture
            return None
        
        return self.gestures_history[-1]
    
    def get_gesture_history(self, gesture_type: GestureType,
                           time_window: float = 2.0) -> List[Gesture]:
        """Get gestures of specific type within time window."""
        current_time = time.time()
        matching = []
        
        for gesture in reversed(self.gestures_history):
            if current_time - gesture.timestamp > time_window:
                break
            
            if gesture.type == gesture_type:
                matching.append(gesture)
        
        return matching
    
    def is_gesture_stable(self, gesture_type: GestureType,
                         time_window: float = 0.5) -> bool:
        """Check if a gesture is held stable over time."""
        history = self.get_gesture_history(gesture_type, time_window)
        return len(history) >= 5  # Arbitrary threshold
    
    def reset(self):
        """Reset gesture engine."""
        self.gestures_history.clear()
        self.last_pinch_time.clear()
