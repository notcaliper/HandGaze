# 🖐 HandGaze - Gesture-Based Text Input System

<div align="center">

![HandGaze Logo](docs/images/logo.png)

[![Python Version](https://img.shields.io/badge/python-3.9.3%20~%203.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.5-orange.svg)](https://mediapipe.dev/)

</div>

## 🚀 Overview

HandGaze is a cutting-edge gesture-based text input system that allows users to type and interact with their computer using hand gestures. By leveraging computer vision and machine learning, HandGaze provides an intuitive and hands-free way to input text.

### ✨ Key Features

- 🎯 Real-time hand gesture recognition
- ⚡️ Fast and responsive text input
- 📝 Smart word suggestions
- 🎨 Modern and intuitive UI
- 🔄 Gesture training system
- 📚 Offline dictionary support
- 📊 Gesture accuracy testing

## 🛠 Installation

### System Requirements

- **Python 3.9.3** (strongly recommended for optimal performance)
  - MediaPipe is compatible with Python 3.9.3 through 3.11
  - We recommend 3.9.3 for best stability
- Webcam or camera device
- Sufficient lighting for hand detection

### Version Notes
- If using Python 3.11, you may need to adjust some MediaPipe dependencies
- For maximum stability, stick with Python 3.9.3
- If you encounter MediaPipe issues with Python 3.11, downgrade to 3.9.3

```bash
# Install Python 3.9.3 (if using pyenv)
pyenv install 3.9.3
pyenv local 3.9.3

# Clone the repository
git clone https://github.com/notcaliper/HandGaze.git
cd HandGaze

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Quick Start

1. **Train Gestures** (Optional - Skip if using pre-trained gestures)
   ```bash
   python gesture_trainer.py
   ```

2. **Run HandGaze**
   ```bash
   python hand_recognition.py
   ```

3. **Test Accuracy** (Optional)
   ```bash
   python gesture_accuracy_tester.py
   ```

## 🎯 Usage

1. **Launch the application**
2. **Position your hand** in front of the camera
3. **Make gestures** corresponding to letters or commands
4. **Hold the gesture** briefly to confirm
5. Use **SPACE** and **BACKSPACE** gestures for word completion

### Performance Tips

- Ensure good lighting conditions
- Keep your hand within the camera frame
- Maintain a stable hand position during gesture recognition
- Use Python 3.9.3 for optimal performance
- Close other applications using the camera

## 🔍 Accuracy Testing

The system includes a built-in accuracy testing tool that:
- Evaluates recognition accuracy for each gesture
- Provides detailed metrics and analysis
- Generates comprehensive test reports
- Helps identify gestures that need retraining

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

## 📚 Documentation

- [Features Documentation](docs/FEATURES.md)
- [Technical Documentation](docs/README.md)
- [Training Guide](docs/TRAINING.md)

## ❗ Troubleshooting

1. **Camera Issues**
   - Check camera permissions
   - Verify no other application is using the camera
   - Ensure proper camera connections

2. **Recognition Problems**
   - Verify Python 3.9.3 installation
   - Check lighting conditions
   - Try retraining gestures
   - Update all dependencies

3. **Performance Issues**
   - Confirm Python 3.9.3 usage
   - Check system resources
   - Close unnecessary applications

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV team for computer vision capabilities
- MediaPipe team for hand tracking technology
- Contributors and community members

---
<div align="center">
Made with ❤️ by HandGaze Team
</div>
