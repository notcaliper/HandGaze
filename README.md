# 🖐 HandGaze - Gesture-Based Text Input System

<div align="center">

![HandGaze Logo](docs/images/logo.png)

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
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

## 🛠 Installation

```bash
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

## 🎯 Usage

1. **Launch the application**
2. **Position your hand** in front of the camera
3. **Make gestures** corresponding to letters or commands
4. **Hold the gesture** briefly to confirm
5. Use **SPACE** and **BACKSPACE** gestures for word completion

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

## 📚 Documentation

- [Features Documentation](docs/FEATURES.md)
- [Technical Documentation](docs/README.md)
- [Training Guide](docs/TRAINING.md)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV team for computer vision capabilities
- MediaPipe team for hand tracking technology
- Contributors and community members

---
<div align="center">
Made with ❤️ by NotCaliper
</div>
