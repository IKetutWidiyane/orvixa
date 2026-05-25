# ORVIXA - Spatial Gesture Workspace

A futuristic real-time computer vision application for spatial hand gesture interaction. ORVIXA allows users to interact with a virtual floating workspace using hand gestures tracked through a webcam.

## 🎯 Features

### Core Capabilities
- **Real-time Hand Tracking** - MediaPipe-based 21-point hand landmark detection
- **Gesture Recognition** - Pinch, pointing, palm, fist, and multi-finger gestures
- **Virtual Air Drawing** - Draw freely in space with brush control
- **Floating Media System** - Load and manipulate images and videos
- **Infinite Virtual Workspace** - Pan, zoom, and navigate large canvas
- **Futuristic UI** - Cyberpunk neon aesthetic with hologram effects
- **Sound Effects** - Immersive audio feedback for interactions

### Gesture Controls
- **One Finger** - Cursor/pointing mode
- **Two Fingers** - Drag mode
- **Pinch** - Zoom in/out
- **Three Fingers** - Rotate objects
- **Open Palm** - Reset view
- **Closed Fist** - Delete selected object

## 🛠 Tech Stack

### Core Libraries
- **OpenCV** - Video capture and image processing
- **MediaPipe** - Hand tracking and pose estimation
- **NumPy** - Numerical computations
- **Pillow** - Image manipulation
- **Pygame** - Sound effects (optional)

### Architecture
- **Python 3.11+**
- Modular component-based design
- Real-time processing pipeline
- Threading support for optimal performance

## 📁 Project Structure

```
ORVIXA/
├── assets/              # Media files
│   ├── images/         # Sample images
│   ├── videos/         # Sample videos
│   ├── sounds/         # Sound effects
│   └── icons/          # UI icons
├── core/               # Core modules
│   ├── camera/         # Video capture
│   ├── tracking/       # Hand tracking
│   ├── interaction/    # Gesture recognition
│   ├── rendering/      # Rendering engine
│   ├── workspace/      # Virtual workspace
│   └── effects/        # Drawing & sound
├── media/              # Media handling
│   ├── image/         # Image management
│   ├── video/         # Video management
│   └── widgets/       # Media widgets
├── ui/                 # User interface
│   ├── themes/        # Theme system
│   ├── panels/        # UI panels
│   ├── animations/    # Animations
│   └── overlays/      # HUD overlays
├── utils/              # Utilities
│   ├── config.py      # Configuration
│   ├── logger.py      # Logging
│   └── geometry.py    # Math utilities
├── saves/              # Saved workspaces
├── tests/              # Test suite
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🚀 Getting Started

### Installation

1. **Clone/Download the project**
```bash
cd ORVIXA
```

2. **Create Python virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running ORVIXA

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `C` | Clear canvas |
| `E` | Toggle eraser |
| `D` | Toggle drawing mode |
| `R` | Reset view |
| `T` | Toggle tool |
| `1` | Load sample media |

## 🎨 Visual Design

### Color Palette
- **Primary**: Neon Blue (`#00D9FF`)
- **Secondary**: Purple (`#8A2BE2`)
- **Accent**: Cyan (`#00FFD9`)
- **Background**: Dark (`#050816`)

### Effects
- Holographic scanlines
- Neon glow borders
- Transparent panels (glassmorphism)
- Smooth animations
- Real-time particle effects

## 🧠 Gesture Recognition

### Hand Landmarks
The system tracks 21 landmarks per hand:
- **Palm** - Center and wrist (0-9)
- **Thumb** - Base to tip (0, 1, 2, 3, 4)
- **Index** - Base to tip (0, 5, 6, 7, 8)
- **Middle** - Base to tip (0, 9, 10, 11, 12)
- **Ring** - Base to tip (0, 13, 14, 15, 16)
- **Pinky** - Base to tip (0, 17, 18, 19, 20)

### Gesture Detection Algorithm

#### Pinch Detection
```
distance = ||thumb_tip - index_tip||
if distance < PINCH_THRESHOLD:
    → PINCH_DETECTED
```

#### Finger State Detection
```
for each finger:
    if tip_y < pip_y:
        → finger_up = True
```

## 🔧 Configuration

Edit `utils/config.py` to customize:
- Camera resolution and FPS
- Hand tracking confidence
- Gesture thresholds
- Workspace dimensions
- Drawing parameters
- Theme colors
- Performance settings

### Key Configuration Values

```python
# Camera
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

# Hand Tracking
HAND_CONFIDENCE = 0.7
MAX_HANDS = 2

# Workspace
WORKSPACE_WIDTH = 4000
WORKSPACE_HEIGHT = 3000
MIN_ZOOM = 0.3
MAX_ZOOM = 3.0

# Drawing
DRAWING_LINE_THICKNESS = 3
DRAWING_COLOR = (0, 217, 255)  # BGR

# Performance
ENABLE_THREADING = True
```

## 🎮 Advanced Features

### Drawing System
- Smooth brush strokes with anti-aliasing
- Color picker support
- Adjustable brush thickness
- Eraser tool
- Canvas clearing

### Media Manipulation
- Drag media objects
- Resize with gestures
- Layer management
- Transparency control
- Video playback

### Workspace Navigation
- Pan camera with gestures
- Zoom with pinch gesture
- Multiple layer support
- Object selection and manipulation

## 🔊 Sound System

Integrated sound effects:
- **click.wav** - UI interactions
- **hologram.wav** - Media loaded
- **zoom.wav** - Zoom gestures
- **notification.wav** - Notifications
- **error.wav** - Errors
- **success.wav** - Success events

## 📊 Performance Optimization

### Threading
- Asynchronous camera capture
- Non-blocking frame processing
- Concurrent gesture detection

### Efficiency Features
- Frame buffering (3-frame buffer)
- Landmark smoothing
- Efficient gesture caching
- Optimized rendering pipeline

### FPS Targets
- **60 FPS** - Smooth interactive experience
- **Minimum 30 FPS** - Comfortable tracking

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Manual Testing Checklist
- [ ] Hand detection on startup
- [ ] Gesture recognition accuracy
- [ ] Drawing smoothness
- [ ] Media loading and display
- [ ] Zoom/pan responsiveness
- [ ] Sound playback
- [ ] UI rendering

## 🐛 Troubleshooting

### Camera Not Detected
- Ensure camera is connected
- Check camera permissions
- Try different camera ID in config

### Poor Hand Tracking
- Improve lighting conditions
- Increase hand detection confidence
- Get hand closer to camera

### Low FPS
- Reduce camera resolution
- Disable threading if CPU-bound
- Close other applications

### No Sound
- Install pygame: `pip install pygame`
- Check sound files in `assets/sounds/`
- Enable sound in config

## 📚 Architecture Overview

```
Input Stream (Webcam)
         ↓
   Camera Capture
         ↓
  Hand Tracking (MediaPipe)
         ↓
 Landmark Detection (21 points)
         ↓
 Gesture Recognition Engine
         ↓
 Interaction Processing
         ↓
 Workspace Rendering
         ↓
  Visual Renderer
         ↓
    UI Overlay
         ↓
  Display Output
```

## 🔄 Processing Pipeline

1. **Capture** - Grab frame from webcam (30 FPS)
2. **Detect** - Find hands and landmarks (MediaPipe)
3. **Recognize** - Identify gestures
4. **Interact** - Process user interactions
5. **Update** - Update workspace state
6. **Render** - Draw workspace and objects
7. **Display** - Show frame with overlays

## 🎓 Learning Resources

### Key Concepts
- MediaPipe Hand Pose Estimation
- Real-time Gesture Recognition
- 2D Graphics Rendering
- Camera Calibration
- Coordinate Transformation

### Documentation
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Guide](https://numpy.org/doc/)

## 🚧 Future Enhancements

### Planned Features
- [ ] 3D gesture recognition
- [ ] Voice commands
- [ ] AI custom gesture training
- [ ] Body segmentation
- [ ] Multi-window system
- [ ] OpenGL rendering
- [ ] CUDA acceleration
- [ ] Hand pose refinement with depth
- [ ] Real-time collaborative mode
- [ ] Gesture recording/playback

### Optimization Opportunities
- GPU acceleration for hand tracking
- Machine learning-based gesture classification
- Advanced gesture combination detection
- Real-time rendering optimization

## 📝 Code Quality

### Design Principles
- **Modular** - Independent, reusable components
- **Scalable** - Easy to extend with new features
- **Testable** - Well-isolated functions
- **Documented** - Clear docstrings and comments
- **Efficient** - Optimized performance-critical paths

### Architecture Patterns
- **Singleton** - Theme and logger management
- **Manager** - Image, video, and media management
- **Component** - Modular core system
- **Pipeline** - Processing chain design
- **Factory** - Object creation

## 📄 License

This project is provided as-is for educational and development purposes.

## 👨‍💻 Author

ORVIXA - Spatial Gesture Workspace System
Developed as a futuristic computer vision application.

## 🤝 Contributing

Contributions welcome! Areas for contribution:
- Gesture recognition improvements
- New UI themes
- Performance optimization
- Documentation
- Test coverage
- Sound effects
- Sample media

---

**ORVIXA** - *Experience the future of spatial computing*
