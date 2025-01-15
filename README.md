# HandGaze 👋 

<div align="center">

![HandGaze Logo](https://img.shields.io/badge/HandGaze-Vision-blue?style=for-the-badge&logo=opencv)

[![Python](https://img.shields.io/badge/python-v3.7+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg?style=for-the-badge&logo=google)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-red.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-GNU_GPL_v3-blue.svg?style=for-the-badge&logo=gnu)](LICENSE)

> *Transform your hand gestures into digital communication with AI-powered recognition* ✨

[Features](docs/FEATURES.md) • [Installation](#-quick-start) • [Usage](#-basic-usage) • [Documentation](docs/README.md)

---

<p align="center">
  <img src="docs/demo.gif" alt="HandGaze Demo" width="600"/>
</p>

</div>

## 🌟 What is HandGaze?

HandGaze is a cutting-edge computer vision application that revolutionizes digital communication through AI-powered hand gesture recognition. Create sentences, type words, and interact with your computer using natural hand movements - all in real-time!

## ✨ Feature Overview

<div align="center">

<table>
<tr>
<td align="center" width="33%">

### 🎯 Core Features

- 95%+ Recognition Accuracy
- 0.8s Response Time
- OpenCL Acceleration
- Real-time Performance
- ASL Gesture Support
- Custom Gesture Training

</td>
<td align="center" width="33%">

### 🔧 Smart Tools

- Predictive Text Input
- Auto-capitalization
- Punctuation Gestures
- Context-aware Corrections
- Word Suggestions
- Dynamic Confidence

</td>
<td align="center" width="33%">

### ⚡ Performance

- Hardware Acceleration
- Smart Caching System
- Optimized Processing
- Low CPU Usage (<20%)
- High FPS Output
- Memory Efficient

</td>
</tr>
</table>

</div>

## 🚀 Quick Start

### 📋 Prerequisites

```bash
Python 3.7+
OpenCV-compatible webcam
4GB RAM minimum
Internet (for initial setup)
```

### ⚡ One-Line Installation

```bash
git clone https://github.com/notcaliper/HandGaze.git && cd HandGaze && pip install -r requirements.txt
```

### 🎮 Basic Usage

1. **Launch HandGaze**
   ```bash
   python hand_recognition.py
   ```

2. **Train Custom Gestures (Optional)**
   ```bash
   python gesture_trainer.py
   ```

3. **Gesture Controls**
   - 🔤 Use ASL gestures for letters
   - 👋 Hold "SPACE" gesture (0.8s) for spaces
   - ✌️ Hold "BACKSPACE" gesture (0.8s) to delete
   - ✊ "SHIFT" gesture for capitalization
   - 👆 "PERIOD" gesture for punctuation

## 💡 Pro Tips

<table>
<tr>
<td width="50%">

### 🎯 For Best Recognition

- Maintain good lighting
- Keep hands within frame
- Make deliberate gestures
- Position camera at eye level
- Use the training mode for custom gestures

</td>
<td width="50%">

### ⚡ For Better Performance

- Close background applications
- Enable hardware acceleration
- Update gesture database regularly
- Use suggested word completions
- Keep hands 2-3 feet from camera

</td>
</tr>
</table>

## 📚 Documentation

<div align="center">

<table>
<tr>
<td align="center" width="25%">

### 📖
[Full Documentation](docs/README.md)

Complete guide

</td>
<td align="center" width="25%">

### ✨
[Features Guide](docs/FEATURES.md)

All features

</td>
<td align="center" width="25%">

### 🎯
[API Reference](docs/README.md#-api-reference)

Technical details

</td>
<td align="center" width="25%">

### 🔧
[Troubleshooting](docs/README.md#-troubleshooting)

Common issues

</td>
</tr>
</table>

</div>

## 📝 License

HandGaze is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [NotCaliper](https://github.com/notcaliper)

</div>
