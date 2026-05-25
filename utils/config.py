"""
Configuration module for ORVIXA.
Centralized settings for camera, rendering, and performance.
"""

# ======================== CAMERA SETTINGS ========================
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

# ======================== HAND TRACKING SETTINGS ========================
HAND_CONFIDENCE = 0.7
MAX_HANDS = 2
STATIC_IMAGE_MODE = False

# ======================== GESTURE SETTINGS ========================
PINCH_THRESHOLD = 0.05  # Distance threshold in normalized coordinates
PINCH_COOLDOWN = 0.3  # Seconds
GESTURE_SMOOTHING = 0.5  # Smoothing factor for hand positions

# ======================== WORKSPACE SETTINGS ========================
WORKSPACE_WIDTH = 4000
WORKSPACE_HEIGHT = 3000
INITIAL_ZOOM = 1.0
MIN_ZOOM = 0.3
MAX_ZOOM = 3.0
PAN_SPEED = 10.0
ZOOM_SPEED = 1.1

# ======================== DRAWING SETTINGS ========================
DRAWING_LINE_THICKNESS = 3
DRAWING_COLOR = (0, 217, 255)  # Neon Blue in BGR
ERASER_SIZE = 50
DRAWING_SMOOTHING = 0.7

# ======================== THEME COLORS ========================
COLORS = {
    'neon_blue': (0, 217, 255),      # BGR
    'purple': (138, 43, 226),         # BGR
    'dark_bg': (5, 8, 22),            # BGR
    'white': (255, 255, 255),
    'dark_gray': (20, 20, 30),
    'green': (0, 255, 0),
    'red': (0, 0, 255),
}

# ======================== PERFORMANCE SETTINGS ========================
ENABLE_THREADING = True
FRAME_BUFFER_SIZE = 3
FPS_DISPLAY = True
RESIZE_FRAME_SCALE = 1.0  # 1.0 = full resolution

# ======================== UI SETTINGS ========================
PANEL_ALPHA = 0.7
PANEL_BLUR = 15
HOLOGRAM_SCANLINE_HEIGHT = 2
HOLOGRAM_SCANLINE_ALPHA = 0.15
GLOW_STRENGTH = 2
ANIMATION_SPEED = 0.05

# ======================== SOUND SETTINGS ========================
SOUND_VOLUME = 0.7
ENABLE_SOUND = True

# ======================== MEDIA SETTINGS ========================
MAX_MEDIA_OBJECTS = 20
MEDIA_MIN_SCALE = 0.1
MEDIA_MAX_SCALE = 5.0
MEDIA_DEFAULT_WIDTH = 400
MEDIA_DEFAULT_HEIGHT = 300

# ======================== PATHS ========================
ASSETS_PATH = "assets"
SOUNDS_PATH = f"{ASSETS_PATH}/sounds"
IMAGES_PATH = f"{ASSETS_PATH}/images"
VIDEOS_PATH = f"{ASSETS_PATH}/videos"
SAVES_PATH = "saves"

# ======================== DEBUG SETTINGS ========================
DEBUG_MODE = False
SHOW_LANDMARKS = False
SHOW_BOUNDING_BOXES = False
SHOW_GESTURE_STATUS = True
