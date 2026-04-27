# 🖐️ HandGaze
### The Future of Gesture-Based Human-Computer Interaction

<div align="center">

![HandGaze Logo](docs/images/logo.png)

**A high-performance, AI-powered gesture recognition system for hands-free text input.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.18-00BFFF?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![MIT License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🚀 Overview

**HandGaze** is a state-of-the-art gesture-recognition pipeline designed to bridge the gap between physical motion and digital input. By combining **MediaPipe's** high-fidelity hand tracking with a custom weighted similarity engine, HandGaze enables real-time, fluid text input without ever touching a keyboard.

Whether for accessibility, sterilized environments, or futuristic interfaces, HandGaze provides a robust and extensible platform for gesture-based control.

---

## ✨ Key Features

*   **🎯 Intelligent Recognition**: Uses a dual-metric similarity engine (joint angles + relative distances) for high-accuracy gesture detection.
*   **💎 Glassmorphic UI**: A stunning, modern interface featuring semi-transparent panels, neon accents, and smooth circular progress indicators.
*   **📝 Smart Autocomplete**: Integrated suggestion engine powered by an optimized offline dictionary for rapid typing.
*   **⚡ High-Performance Pipeline**: Multi-threaded camera handling and optimized AI inference for 30+ FPS performance.
*   **🔄 Gesture Trainer**: A built-in utility to train and personalize the system for your own unique hand gestures.
*   **📚 Optimized Dictionary**: Custom-formatted 370k+ word dictionary optimized for low memory footprint and fast lookup.

---

## 🎨 UI Design Philosophy

HandGaze 2.0 features a **Cyber-Industrial** aesthetic:
- **Glassmorphism**: Frosted panels that blend seamlessly with the camera feed.
- **Dynamic Feedback**: Interactive hand bounding boxes with neon corner brackets.
- **Circular Progress**: Intuitive gesture confirmation rings that prevent accidental inputs.
- **Neural Splash**: High-tech initialization sequence for a premium start-up experience.

---

## 🛠️ Technology Stack

- **Core Logic**: Python 3.11+
- **Computer Vision**: OpenCV
- **AI/ML Model**: MediaPipe (Hands v0.10)
- **Natural Language**: PySpellChecker (for smart suggestions)
- **Data Persistence**: Pickle-based gesture profiles

---

## 🎮 Getting Started

### 1. Prerequisites
Ensure you have Python 3.11 or higher installed.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/notcaliper/HandGaze.git
cd HandGaze

# Setup virtual environment
python -m venv venv
./venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App
```bash
# Launch the main recognition system
python hand_recognition.py
```

---

## 🎯 How to Use

1.  **Initialization**: Allow the "Neural Pipeline" to initialize your camera.
2.  **Positioning**: Place your hand within the camera's view.
3.  **Gestures**:
    *   Perform a letter gesture (A-Z).
    *   Hold the gesture until the **Circular Ring** completes (100%).
    *   Use the **SPACE** gesture (closed fist) to complete a word.
    *   Use **BACKSPACE** (open palm sideways) to delete.
4.  **Suggestions**: The top 3 suggestions appear in the text panel. Hold the corresponding gesture to select.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for feature requests.

---

<div align="center">
Made by <b>NotCaliper</b>
</div>
