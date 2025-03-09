# HandGaze v1.0.0 - Initial Stable Release

## 🚀 Release Highlights

This is the first stable release of HandGaze, a gesture-based text input system that allows users to type and interact with their computer using hand gestures.

## ✨ Key Features

- 🎯 Real-time hand gesture recognition
- ⚡️ Fast and responsive text input
- 📝 Smart word suggestions
- 🎨 Modern and intuitive UI
- 🔄 Gesture training system
- 📚 Offline dictionary support
- 🛡️ Robust error handling and state management
- 🔄 Automatic system recovery

## 🐛 Bug Fixes

- Enhanced stability in low-light conditions
- Improved hand detection when partially visible
- Fixed gesture recognition cooldown timing
- Corrected spelling suggestions in the dictionary
- Resolved UI flicker during state transitions

## 🔧 Technical Notes

- Framework: Python 3.11 with OpenCV 4.8.1 and MediaPipe 0.10.18
- Performance: 30+ FPS on mid-range hardware
- Recognition accuracy: >95% in good lighting conditions

## 📝 Installation Instructions

```bash
# Clone the repository
git clone https://github.com/notcaliper/HandGaze.git
cd HandGaze

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run HandGaze
python hand_recognition.py
```

## 🙏 Acknowledgments

Thanks to all beta testers and contributors who helped make this release possible!

---

HandGaze is released under the MIT License. 