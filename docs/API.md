# 📖 HandGaze API Documentation

## Overview

This document provides comprehensive API documentation for HandGaze's core classes and functions.

## 🏗️ Core Classes

### `CustomHandGestureRecognizer`

The main class for gesture recognition and text input.

#### Constructor

```python
def __init__(self):
    """
    Initialize the gesture recognition system.
    
    Raises:
        RuntimeError: If camera cannot be opened
        FileNotFoundError: If gesture data is not found
    """
```

#### Methods

##### `recognize_gesture(landmarks)`

Recognizes a gesture from hand landmarks using advanced algorithms.

```python
def recognize_gesture(self, landmarks) -> str:
    """
    Recognize gesture from MediaPipe hand landmarks.
    
    Args:
        landmarks: MediaPipe hand landmarks object
        
    Returns:
        str: Recognized gesture name or "Unknown"
        
    Example:
        >>> recognizer = CustomHandGestureRecognizer()
        >>> gesture = recognizer.recognize_gesture(hand_landmarks)
        >>> print(gesture)  # "A" or "B" or "Unknown"
    """
```

##### `calculate_angles(landmarks)`

Calculate joint angles between finger segments.

```python
def calculate_angles(self, landmarks) -> List[float]:
    """
    Calculate 15+ joint angles for gesture classification.
    
    Args:
        landmarks: List of [x, y] coordinates for hand landmarks
        
    Returns:
        List[float]: List of joint angles in radians
        
    Example:
        >>> angles = recognizer.calculate_angles(landmark_coords)
        >>> print(f"Thumb angle: {angles[0]:.2f} radians")
    """
```

##### `get_relative_distances(landmarks)`

Calculate relative distances between key finger points.

```python
def get_relative_distances(self, landmarks) -> np.ndarray:
    """
    Calculate relative distances between fingertips.
    
    Args:
        landmarks: List of [x, y] coordinates for hand landmarks
        
    Returns:
        np.ndarray: Array of relative distances
        
    Example:
        >>> distances = recognizer.get_relative_distances(landmarks)
        >>> print(f"Thumb-index distance: {distances[0]:.3f}")
    """
```

##### `add_gesture_to_text(gesture)`

Add recognized gesture to the text buffer with dictionary support.

```python
def add_gesture_to_text(self, gesture: str) -> None:
    """
    Process gesture and add to text with smart dictionary features.
    
    Args:
        gesture: Recognized gesture name
        
    Side Effects:
        - Updates self.current_word
        - Updates self.sentence
        - Updates self.word_suggestions
        
    Example:
        >>> recognizer.add_gesture_to_text("H")
        >>> recognizer.add_gesture_to_text("E")
        >>> recognizer.add_gesture_to_text("SPACE")
    """
```

##### `process_frame(frame)`

Process a video frame for gesture recognition and UI rendering.

```python
def process_frame(self, frame) -> np.ndarray:
    """
    Process video frame for gesture recognition and UI rendering.
    
    Args:
        frame: OpenCV BGR image frame
        
    Returns:
        np.ndarray: Processed frame with UI elements
        
    Example:
        >>> ret, frame = cap.read()
        >>> processed = recognizer.process_frame(frame)
        >>> cv2.imshow('HandGaze', processed)
    """
```

##### `run()`

Main execution loop for the gesture recognition system.

```python
def run(self) -> None:
    """
    Start the main gesture recognition loop.
    
    Handles:
        - Camera capture
        - Frame processing
        - User input
        - Resource cleanup
        
    Example:
        >>> recognizer = CustomHandGestureRecognizer()
        >>> recognizer.run()  # Starts the application
    """
```

#### Properties

##### `current_word`
```python
@property
def current_word(self) -> str:
    """Current word being typed."""
```

##### `sentence`
```python
@property
def sentence(self) -> str:
    """Complete sentence with finished words."""
```

##### `word_suggestions`
```python
@property
def word_suggestions(self) -> List[str]:
    """List of word suggestions for current input."""
```

##### `fps`
```python
@property
def fps(self) -> float:
    """Current frames per second."""
```

---

### `GestureTrainer`

Interactive training system for custom gestures.

#### Constructor

```python
def __init__(self):
    """
    Initialize the gesture training system.
    
    Creates necessary directories and loads existing data.
    """
```

#### Methods

##### `record_samples(gesture_name, num_samples)`

Record multiple samples for a gesture.

```python
def record_samples(self, gesture_name: str, num_samples: int = 5) -> bool:
    """
    Record multiple samples for robust gesture training.
    
    Args:
        gesture_name: Name of the gesture to train
        num_samples: Number of samples to record
        
    Returns:
        bool: True if training successful
        
    Example:
        >>> trainer = GestureTrainer()
        >>> success = trainer.record_samples("HELLO", 5)
        >>> print(f"Training {'successful' if success else 'failed'}")
    """
```

##### `validate_gesture(gesture_name)`

Validate a trained gesture with confidence scoring.

```python
def validate_gesture(self, gesture_name: str) -> float:
    """
    Validate trained gesture and return confidence score.
    
    Args:
        gesture_name: Name of gesture to validate
        
    Returns:
        float: Confidence score (0.0 to 1.0)
        
    Example:
        >>> confidence = trainer.validate_gesture("A")
        >>> print(f"Gesture confidence: {confidence:.2f}")
    """
```

##### `backup_data()`

Create backup of gesture data with timestamp.

```python
def backup_data(self) -> str:
    """
    Create timestamped backup of gesture data.
    
    Returns:
        str: Path to backup file
        
    Example:
        >>> backup_path = trainer.backup_data()
        >>> print(f"Data backed up to: {backup_path}")
    """
```

##### `get_gesture_summary()`

Get summary of trained gestures.

```python
def get_gesture_summary(self) -> Dict[str, int]:
    """
    Get summary of all trained gestures.
    
    Returns:
        Dict[str, int]: Mapping of gesture names to sample counts
        
    Example:
        >>> summary = trainer.get_gesture_summary()
        >>> for gesture, count in summary.items():
        ...     print(f"{gesture}: {count} samples")
    """
```

##### `export_gestures(filename)`

Export gesture data to file.

```python
def export_gestures(self, filename: str) -> bool:
    """
    Export gesture data to specified file.
    
    Args:
        filename: Path to export file
        
    Returns:
        bool: True if export successful
        
    Example:
        >>> trainer.export_gestures("my_gestures.pkl")
    """
```

##### `import_gestures(filename)`

Import gesture data from file.

```python
def import_gestures(self, filename: str) -> bool:
    """
    Import gesture data from file.
    
    Args:
        filename: Path to import file
        
    Returns:
        bool: True if import successful
        
    Example:
        >>> trainer.import_gestures("downloaded_gestures.pkl")
    """
```

---

### `OfflineDictionary`

Smart dictionary system with spell correction and suggestions.

#### Constructor

```python
def __init__(self):
    """
    Initialize the offline dictionary system.
    
    Loads or creates dictionary data.
    """
```

#### Methods

##### `get_suggestions(word)`

Get spelling suggestions for a word.

```python
def get_suggestions(self, word: str) -> List[str]:
    """
    Get context-aware word suggestions.
    
    Args:
        word: Input word (possibly misspelled)
        
    Returns:
        List[str]: List of up to 3 suggested words
        
    Example:
        >>> dictionary = OfflineDictionary()
        >>> suggestions = dictionary.get_suggestions("helo")
        >>> print(suggestions)  # ["hello", "help", "held"]
    """
```

##### `is_valid_word(word)`

Check if a word is valid.

```python
def is_valid_word(self, word: str) -> bool:
    """
    Check if word exists in dictionary.
    
    Args:
        word: Word to check
        
    Returns:
        bool: True if word is valid
        
    Example:
        >>> dictionary.is_valid_word("hello")  # True
        >>> dictionary.is_valid_word("helo")   # False
    """
```

##### `add_word(word, info)`

Add custom word to dictionary.

```python
def add_word(self, word: str, info: Dict = None) -> None:
    """
    Add custom word to dictionary.
    
    Args:
        word: Word to add
        info: Optional word metadata
        
    Example:
        >>> dictionary.add_word("tensorflow", {"frequency": 100})
    """
```

##### `get_word_info(word)`

Get information about a word.

```python
def get_word_info(self, word: str) -> Optional[Dict]:
    """
    Get metadata for a word.
    
    Args:
        word: Word to look up
        
    Returns:
        Optional[Dict]: Word metadata or None
        
    Example:
        >>> info = dictionary.get_word_info("hello")
        >>> print(info["frequency"])  # Usage frequency
    """
```

##### `update_frequency(word, delta)`

Update word usage frequency.

```python
def update_frequency(self, word: str, delta: int = 1) -> None:
    """
    Update word usage frequency for better suggestions.
    
    Args:
        word: Word to update
        delta: Frequency change (default: +1)
        
    Example:
        >>> dictionary.update_frequency("python", 5)
    """
```

---

## 🔧 Utility Functions

### `load_gesture_data(filename)`

Load gesture data from file.

```python
def load_gesture_data(filename: str) -> Dict:
    """
    Load gesture data from pickle file.
    
    Args:
        filename: Path to gesture data file
        
    Returns:
        Dict: Loaded gesture data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Example:
        >>> data = load_gesture_data("gesture_data.pkl")
        >>> print(f"Loaded {len(data)} gestures")
    """
```

### `save_gesture_data(data, filename)`

Save gesture data to file.

```python
def save_gesture_data(data: Dict, filename: str) -> None:
    """
    Save gesture data to pickle file.
    
    Args:
        data: Gesture data dictionary
        filename: Path to save file
        
    Example:
        >>> save_gesture_data(my_gestures, "backup.pkl")
    """
```

### `calculate_fps(timestamps)`

Calculate frames per second from timestamps.

```python
def calculate_fps(timestamps: List[float]) -> float:
    """
    Calculate FPS from list of timestamps.
    
    Args:
        timestamps: List of frame timestamps
        
    Returns:
        float: Calculated FPS
        
    Example:
        >>> fps = calculate_fps([0.0, 0.033, 0.066, 0.099])
        >>> print(f"FPS: {fps:.1f}")  # ~30.0
    """
```

---

## 🎯 Configuration Constants

### Recognition Settings

```python
# Gesture recognition thresholds
CONFIDENCE_THRESHOLD = 0.25    # Lower = more strict recognition
GESTURE_DELAY = 1.0           # Seconds to hold gesture
PROCESS_EVERY_N_FRAMES = 2    # Frame skipping for performance

# Buffer sizes
GESTURE_BUFFER_SIZE = 3       # Frames for gesture smoothing
FPS_BUFFER_SIZE = 10         # FPS calculation window
```

### Camera Settings

```python
# Camera configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1       # Minimize frame buffer
```

### UI Settings

```python
# Display configuration
MAX_CHARS_PER_LINE = 40      # Text wrapping
FONT_SCALE = 0.7            # Text size
FONT_THICKNESS = 2          # Text thickness
```

---

## 📊 Data Structures

### Gesture Data Format

```python
gesture_data = {
    "A": [
        [[x1, y1], [x2, y2], ...],  # Sample 1 (21 landmarks)
        [[x1, y1], [x2, y2], ...],  # Sample 2
        # ... more samples
    ],
    "B": [
        # ... samples for gesture B
    ]
}
```

### Word Info Format

```python
word_info = {
    "word": "hello",
    "frequency": 1000,
    "category": "common",
    "added_by": "user"
}
```

### Performance Metrics

```python
metrics = {
    "fps": 30.5,
    "cpu_usage": 0.20,
    "memory_usage": 185.7,  # MB
    "recognition_accuracy": 0.95,
    "latency": 0.045  # seconds
}
```

---

## 🚀 Usage Examples

### Basic Recognition

```python
from hand_recognition import CustomHandGestureRecognizer

# Initialize recognizer
recognizer = CustomHandGestureRecognizer()

# Start recognition
recognizer.run()
```

### Custom Training

```python
from gesture_trainer import GestureTrainer

# Initialize trainer
trainer = GestureTrainer()

# Train new gesture
trainer.record_samples("THUMBS_UP", 5)

# Validate training
confidence = trainer.validate_gesture("THUMBS_UP")
print(f"Training confidence: {confidence:.2f}")
```

### Dictionary Operations

```python
from offline_dictionary import OfflineDictionary

# Initialize dictionary
dictionary = OfflineDictionary()

# Add custom words
dictionary.add_word("HandGaze", {"frequency": 50})

# Get suggestions
suggestions = dictionary.get_suggestions("helo")
print(suggestions)  # ["hello", "help", "held"]
```

### Advanced Configuration

```python
# Custom recognition with modified settings
recognizer = CustomHandGestureRecognizer()
recognizer.confidence_threshold = 0.2  # More strict
recognizer.gesture_delay = 0.5         # Faster response
recognizer.process_every_n_frames = 1  # Process every frame

# Custom training parameters
trainer = GestureTrainer()
trainer.min_samples = 3
trainer.validation_threshold = 0.8
```

---

## 🛠️ Error Handling

### Common Exceptions

```python
# Camera initialization errors
try:
    recognizer = CustomHandGestureRecognizer()
except RuntimeError as e:
    print(f"Camera error: {e}")

# Gesture data errors
try:
    recognizer.run()
except FileNotFoundError:
    print("Please train gestures first!")

# Training errors
try:
    trainer.record_samples("TEST", 5)
except ValueError as e:
    print(f"Training error: {e}")
```

### Error Recovery

```python
# Automatic recovery patterns
def safe_recognize_gesture(recognizer, landmarks):
    try:
        return recognizer.recognize_gesture(landmarks)
    except Exception as e:
        print(f"Recognition error: {e}")
        return "Unknown"

# Resource cleanup
def cleanup_resources(recognizer):
    recognizer.cap.release()
    cv2.destroyAllWindows()
    recognizer.hands.close()
```

---

This API documentation provides comprehensive coverage of HandGaze's functionality. For more examples and advanced usage, see the main documentation and source code.