# HandGaze 👋 

<div align="center">

![HandGaze Logo](https://img.shields.io/badge/HandGaze-Vision-blue?style=for-the-badge&logo=opencv)

[![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> *Transform your hand gestures into digital communication with AI-powered recognition* ✨

[Features](docs/FEATURES.md) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](docs/README.md)

---

<p align="center">
  <img src="https://raw.githubusercontent.com/notcaliper/HandGaze/main/docs/demo.gif" alt="HandGaze Demo" width="600"/>
</p>

</div>

## 📚 Quick Documentation

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

## 🌟 What is HandGaze?

HandGaze is a cutting-edge computer vision application that revolutionizes digital communication through AI-powered hand gesture recognition. Create sentences, type words, and interact with your computer using natural hand movements - all in real-time!

<div align="center">

```mermaid
graph LR
    A[👁️ Camera Input] --> B[🤚 Hand Detection]
    B --> C[📍 Landmark Tracking]
    C --> D[🎯 Gesture Recognition]
    D --> E[💭 Word Processing]
    E --> F[📝 Sentence Creation]
    style A fill:#ff9999,stroke:#ff0000,stroke-width:2px,color:#990000,font-weight:bold
    style B fill:#99ff99,stroke:#00ff00,stroke-width:2px,color:#006600,font-weight:bold
    style C fill:#9999ff,stroke:#0000ff,stroke-width:2px,color:#000099,font-weight:bold
    style D fill:#ffff99,stroke:#ffff00,stroke-width:2px,color:#999900,font-weight:bold
    style E fill:#ff99ff,stroke:#ff00ff,stroke-width:2px,color:#990099,font-weight:bold
    style F fill:#99ffff,stroke:#00ffff,stroke-width:2px,color:#009999,font-weight:bold
    linkStyle default stroke:#333333,stroke-width:2px
```

</div>

## ✨ Features

<div align="center">

| Core Features | Smart Features | User Experience |
|--------------|----------------|-----------------|
| 🎯 Real-time Recognition | 🔍 Word Suggestions | 🎨 Interactive UI |
| 🤚 Hand Tracking | 📝 Sentence Building | ⚡ High Performance |
| 🔤 ASL Support | ⌫ Smart Backspace | 📊 Visual Feedback |
| ✨ Custom Gestures | 💡 Auto-Correction | 🎮 Gesture Control |

[View Full Features Guide](docs/FEATURES.md) 📚

</div>

## 🎯 Latest Updates (v2.0)

<table>
<tr>
<td width="50%">

### ✨ New Features

- **Sentence Creation**
  - Build complete sentences
  - Smart word spacing
  - Line wrapping support
  
- **Enhanced Text Editing**
  - Context-aware backspace
  - Word suggestion system
  - Faster gesture response (1.5s)

</td>
<td width="50%">

### 🛠️ Technical Updates

- **Performance**
  - Optimized recognition engine
  - Improved memory usage
  - Enhanced error handling

- **User Experience**
  - Better visual feedback
  - Smoother animations
  - Real-time suggestions

</td>
</tr>
</table>

## 🚀 Quick Start

### 📋 Prerequisites

```bash
Python 3.7+
Webcam
Internet (for initial setup)
```

### ⚡ One-Line Installation

```bash
git clone https://github.com/notcaliper/HandGaze.git && cd HandGaze && pip install -r requirements.txt
```

### 🎮 Basic Usage

1. **Start HandGaze**
   ```bash
   python hand_recognition.py
   ```

2. **Gesture Controls**
   - 🔤 Use ASL gestures for letters
   - 👋 Hold "SPACE" gesture (1.5s) for spaces
   - ✌️ Hold "BACKSPACE" gesture (1.5s) to delete

## 💡 Pro Tips

<table>
<tr>
<td width="50%">

### 🎯 For Best Recognition

- Keep hands within frame
- Use good lighting
- Make clear gestures
- Stay in camera view
- Watch the hold timer

</td>
<td width="50%">

### ⚡ For Better Performance

- Use suggested words
- Practice common gestures
- Keep steady hand position
- Use word predictions
- Follow visual feedback

</td>
</tr>
</table>

## 🛠️ Project Structure

```
HandGaze/
├── 📜 hand_recognition.py  # Main recognition system
├── 🎯 object_detector.py   # Object detection
├── 📚 offline_dictionary.py # Word suggestions
├── ⚙️ gesture_trainer.py   # Custom gesture training
├── 📋 requirements.txt     # Dependencies
└── 📁 data/
    ├── dictionary_data/    # Word database
    └── gesture_data/       # Trained gestures
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📚 Improve documentation

## 📄 License

HandGaze is GNU GPLv3 licensed. See [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ by [NotCaliper](https://github.com/notcaliper)

</div>
