# HandGaze Documentation 📚

<div align="center">

[![Documentation](https://img.shields.io/badge/HandGaze-Documentation-blue?style=for-the-badge&logo=opencv)](https://github.com/notcaliper/HandGaze)
[![Version](https://img.shields.io/badge/Version-2.1-green?style=for-the-badge)](https://github.com/notcaliper/HandGaze/releases)
[![Python](https://img.shields.io/badge/Python-3.7+-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red?style=for-the-badge&logo=opencv)](https://opencv.org/)

*Your comprehensive guide to mastering HandGaze* 🌟

</div>

## 📑 Table of Contents

- [Installation Guide](#-installation-guide)
- [Core Components](#-core-components)
- [Usage Guide](#-usage-guide)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)

## 📥 Installation Guide

### System Requirements

```yaml
Minimum:
  CPU: Dual-core 2GHz+
  RAM: 4GB
  Camera: OpenCV-compatible webcam
  Python: 3.7+
  Storage: 500MB free space

Recommended:
  CPU: Quad-core 3GHz+
  RAM: 8GB+
  Camera: HD Webcam (1080p)
  Python: 3.9+
  GPU: OpenCL compatible
```

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/notcaliper/HandGaze.git
   cd HandGaze
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Core Components

### 1. Hand Recognition System (v2.1)
- Enhanced real-time hand tracking (95%+ accuracy)
- Multi-hand support with dynamic switching
- Optimized MediaPipe integration
- Advanced gesture prediction
- 0.8s response time

### 2. Gesture Trainer
- Interactive training interface with real-time feedback
- Dynamic confidence indicators
- Automated gesture validation
- Performance metrics tracking
- Cross-validation testing

### 3. Dictionary System
- Predictive text suggestions
- Auto-capitalization support
- Punctuation gesture recognition
- Context-aware corrections
- Efficient word frequency analysis

## 🎮 Usage Guide

### Basic Controls

- 🔤 ASL gestures for letters
- 👋 SPACE gesture (0.8s hold)
- ✌️ BACKSPACE gesture (0.8s hold)
- ✊ SHIFT gesture for capitalization
- 👆 PERIOD gesture for punctuation

### Training New Gestures

1. Launch the trainer:
   ```bash
   python gesture_trainer.py
   ```

2. Select "Add new gesture"
3. Position hand 2-3 feet from camera
4. Press SPACE to capture samples (minimum 20)
5. Use 'r' to redo last sample
6. Press 'q' to save and exit

### Using the Main Application

1. Start HandGaze:
   ```bash
   python hand_recognition.py
   ```

2. Perform gestures within camera frame
3. Watch confidence indicators
4. Use predictive text suggestions
5. Enable hardware acceleration if available

## 🔍 API Reference

### HandRecognition Class
```python
class HandRecognition:
    def __init__(self, use_gpu: bool = True)
    def process_frame(frame: np.ndarray) -> Dict[str, Any]
    def get_gesture_confidence(gesture: str) -> float
    def enable_multihand(enabled: bool = True) -> None
```

### GestureTrainer Class
```python
class GestureTrainer:
    def __init__(self, data_dir: str = "gesture_data")
    def capture_gesture(gesture_name: str, samples: int = 20) -> bool
    def validate_gesture(gesture_name: str) -> float
    def export_metrics(path: str) -> Dict[str, Any]
```

### OfflineDictionary Class
```python
class OfflineDictionary:
    def __init__(self, lang: str = "en")
    def get_predictions(context: str) -> List[str]
    def auto_correct(word: str) -> str
    def add_custom_word(word: str, frequency: int = 1) -> None
```

## 🔧 Troubleshooting

### Performance Optimization

1. **Enable Hardware Acceleration**
   - Check GPU compatibility
   - Update graphics drivers
   - Enable OpenCL support

2. **Improve Recognition**
   - Maintain 2-3 feet distance
   - Use consistent lighting
   - Keep steady hand position
   - Regular gesture retraining

3. **Memory Management**
   - Close background applications
   - Monitor RAM usage
   - Clear gesture cache if needed

### Error Recovery

- Automatic gesture database backup
- Recovery mode for corrupted data
- Fallback to CPU processing
- Diagnostic logging system

## 💡 Pro Tips

1. **For Best Accuracy**
   - Train in various lighting conditions
   - Use deliberate, distinct gestures
   - Keep consistent hand orientation
   - Monitor confidence metrics

2. **For Better Performance**
   - Enable hardware acceleration
   - Update gesture database regularly
   - Use suggested word completions
   - Keep hands in optimal range

## 🤝 Support

- Check [Troubleshooting Guide](#-troubleshooting)
- Review [API Reference](#-api-reference)
- Submit issues on GitHub

## 📝 License 

HandGaze is GNU GPLv3 licensed. See [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ by [NotCaliper](https://github.com/notcaliper)

</div>
