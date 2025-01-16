import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import pickle
import os
from typing import List, Dict, Optional
import tensorflow as tf
from offline_dictionary import OfflineDictionary

class CustomHandGestureRecognizer:
    def __init__(self):
        # Initialize TensorFlow for computation
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("TensorFlow using GPU:", gpus[0].name)
            # Allow memory growth to prevent TF from taking all GPU memory
            tf.config.experimental.set_memory_growth(gpus[0], True)
        else:
            print("TensorFlow using CPU")
        
        # Initialize video capture with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        
        # Initialize MediaPipe Hands with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load gesture data
        self.gesture_data = self.load_latest_gesture_data()
        if not self.gesture_data:
            print("No gesture data found! Please run gesture_trainer.py first.")
            raise FileNotFoundError("No gesture data available")
        
        # Pre-compute gesture features for faster comparison
        self.precomputed_gesture_features = self.precompute_gesture_features()
        
        # Performance monitoring
        self.fps_buffer = deque(maxlen=10)  # Reduced buffer size
        self.prev_frame_time = 0
        
        # Gesture smoothing
        self.gesture_buffer = deque(maxlen=3)  # Reduced buffer size
        
        # Frame processing optimization
        self.process_every_n_frames = 2
        self.frame_count = 0
        self.last_gesture = "Unknown"
        
        # Initialize offline dictionary
        self.dictionary_helper = OfflineDictionary()
        
        # Add word building variables
        self.current_word = ""
        self.word_suggestions: List[str] = []
        self.show_suggestions = False
        
        # Add gesture typing variables
        self.typed_text = ""
        self.last_gesture_time = time.time()
        self.gesture_delay = 1.5  # Reduced from 3.0 to 1.5 seconds for better responsiveness
        self.current_gesture = "Unknown"
        self.gesture_confirmed = False
        self.sentence = ""  # Add sentence variable to store complete text
        
    def load_latest_gesture_data(self):
        """Load the most recent gesture data file"""
        data_dir = 'gesture_data'
        if not os.path.exists(data_dir):
            return None
            
        # Find the latest gesture data file
        if os.path.exists(os.path.join(data_dir, 'gesture_data.pkl')):
            with open(os.path.join(data_dir, 'gesture_data.pkl'), 'rb') as f:
                return pickle.load(f)
                
        files = [f for f in os.listdir(data_dir) if f.startswith('gesture_data_') and f.endswith('.pkl')]
        if not files:
            return None
            
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
        with open(os.path.join(data_dir, latest_file), 'rb') as f:
            return pickle.load(f)
    
    def precompute_gesture_features(self):
        """Pre-compute features for all gesture samples"""
        precomputed = {}
        for gesture_name, gesture_samples in self.gesture_data.items():
            precomputed[gesture_name] = []
            for sample in gesture_samples:
                sample_2d = [[p[0], p[1]] for p in sample]
                angles = self.calculate_angles(sample_2d)
                rel_distances = self.get_relative_distances(sample_2d)
                precomputed[gesture_name].append({
                    'angles': angles,
                    'rel_distances': rel_distances
                })
        return precomputed

    def get_relative_distances(self, landmarks):
        """Calculate relative distances between key points using TensorFlow"""
        key_points = [4, 8, 12, 16, 20]
        n_distances = (len(key_points) * (len(key_points) - 1)) // 2
        distances = tf.zeros(n_distances, dtype=tf.float32)
        
        # Convert landmarks to tensor
        landmarks_tensor = tf.convert_to_tensor(landmarks, dtype=tf.float32)
        
        idx = 0
        for i in range(len(key_points)):
            p1 = tf.convert_to_tensor(landmarks[key_points[i]], dtype=tf.float32)
            for j in range(i + 1, len(key_points)):
                p2 = tf.convert_to_tensor(landmarks[key_points[j]], dtype=tf.float32)
                dist = tf.sqrt(tf.reduce_sum(tf.square(p1 - p2)))
                distances = tf.tensor_scatter_nd_update(distances, [[idx]], [dist])
                idx += 1
                
        return distances.numpy()

    def calculate_angles(self, landmarks):
        """Calculate angles between finger joints using TensorFlow"""
        angles = []
        # Define finger joint connections (indices)
        finger_joints = [
            [0, 1, 2], [1, 2, 3], [2, 3, 4],  # Thumb
            [0, 5, 6], [5, 6, 7], [6, 7, 8],  # Index
            [0, 9, 10], [9, 10, 11], [10, 11, 12],  # Middle
            [0, 13, 14], [13, 14, 15], [14, 15, 16],  # Ring
            [0, 17, 18], [17, 18, 19], [18, 19, 20]  # Pinky
        ]
        
        # Convert landmarks to TensorFlow tensor
        landmarks_tensor = tf.convert_to_tensor(landmarks, dtype=tf.float32)
        
        for p1, p2, p3 in finger_joints:
            v1 = tf.convert_to_tensor([
                landmarks[p1][0] - landmarks[p2][0],
                landmarks[p1][1] - landmarks[p2][1]
            ], dtype=tf.float32)
            
            v2 = tf.convert_to_tensor([
                landmarks[p3][0] - landmarks[p2][0],
                landmarks[p3][1] - landmarks[p2][1]
            ], dtype=tf.float32)
            
            # Normalize vectors
            v1_norm = tf.norm(v1)
            v2_norm = tf.norm(v2)
            
            if tf.equal(v1_norm, 0.0) or tf.equal(v2_norm, 0.0):
                angles.append(0.0)
                continue
            
            v1_normalized = v1 / v1_norm
            v2_normalized = v2 / v2_norm
            
            # Calculate angle
            dot_product = tf.tensordot(v1_normalized, v2_normalized, axes=1)
            angle = tf.acos(tf.clip_by_value(dot_product, -1.0, 1.0))
            angles.append(float(angle.numpy()))
            
        return angles

    def calculate_landmark_similarity(self, landmarks1, landmarks2_features):
        """Optimized similarity calculation using TensorFlow"""
        # Convert inputs to TensorFlow tensors
        angles1 = tf.convert_to_tensor(self.calculate_angles(landmarks1), dtype=tf.float32)
        rel_dist1 = tf.convert_to_tensor(self.get_relative_distances(landmarks1), dtype=tf.float32)
        
        angles2 = tf.convert_to_tensor(landmarks2_features['angles'], dtype=tf.float32)
        rel_dist2 = tf.convert_to_tensor(landmarks2_features['rel_distances'], dtype=tf.float32)
        
        # Calculate differences using TensorFlow
        angle_diff = tf.reduce_mean(tf.abs(angles1 - angles2))
        rel_dist_diff = tf.reduce_mean(tf.abs(rel_dist1 - rel_dist2))
        
        # Combine similarities with weights
        return float((angle_diff * 0.6 + rel_dist_diff * 0.4).numpy())

    def recognize_gesture(self, current_landmarks):
        """Optimized gesture recognition"""
        if not current_landmarks:
            return "Unknown"
        
        # Process only every nth frame
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_gesture
            
        # Convert current landmarks to list format
        current_landmarks_list = [[lm.x, lm.y] for lm in current_landmarks.landmark]
        
        # Compare with pre-computed features
        best_score = float('inf')
        best_gesture = "Unknown"
        
        for gesture_name, gesture_features in self.precomputed_gesture_features.items():
            # Compare with only the first few samples for each gesture
            scores = [self.calculate_landmark_similarity(current_landmarks_list, features) 
                     for features in gesture_features[:3]]  # Limit to first 3 samples
            avg_score = np.mean(scores)
            
            if avg_score < best_score:
                best_score = avg_score
                best_gesture = gesture_name
        
        if best_score > 0.25:
            best_gesture = "Unknown"
        
        self.last_gesture = best_gesture
        return best_gesture

    def add_gesture_to_text(self, gesture: str) -> None:
        """
        Add recognized gesture to typed text with delay and dictionary support
        """
        current_time = time.time()
        
        # Only process if it's a valid gesture (not Unknown)
        if gesture != "Unknown":
            # If this is a new gesture
            if gesture != self.current_gesture:
                self.current_gesture = gesture
                self.gesture_confirmed = False
                self.last_gesture_time = current_time
            # If same gesture is held for required delay
            elif not self.gesture_confirmed and (current_time - self.last_gesture_time) >= self.gesture_delay:
                if gesture == "SPACE":
                    # Add current word to sentence with space
                    if self.current_word:
                        if not self.dictionary_helper.is_valid_word(self.current_word):
                            suggestions = self.dictionary_helper.get_suggestions(self.current_word)
                            if suggestions:
                                self.current_word = suggestions[0]
                        self.sentence += self.current_word + " "
                        self.current_word = ""
                        self.word_suggestions = []
                elif gesture == "BACKSPACE":
                    if self.current_word:
                        # If there's a current word, delete its last character
                        self.current_word = self.current_word[:-1]
                        if self.current_word:
                            self.word_suggestions = self.dictionary_helper.get_suggestions(self.current_word)[:3]
                    elif self.sentence:
                        # If no current word but sentence exists, remove last character from sentence
                        self.sentence = self.sentence[:-1]
                        # If we removed a space, move the last word to current_word for editing
                        if self.sentence and self.sentence[-1] == " ":
                            words = self.sentence.strip().split()
                            if words:
                                self.current_word = words[-1]
                                self.sentence = " ".join(words[:-1]) + " " if len(words) > 1 else ""
                                self.word_suggestions = self.dictionary_helper.get_suggestions(self.current_word)[:3]
                else:
                    self.current_word += gesture
                    # Get suggestions for current word
                    if len(self.current_word) >= 2:
                        self.word_suggestions = self.dictionary_helper.get_suggestions(self.current_word)[:3]
                
                self.gesture_confirmed = True
                print(f"Current word: {self.current_word}")
                print(f"Suggestions: {self.word_suggestions}")
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Display complete sentence and current word
        display_text = self.sentence + self.current_word
        # Split text into lines if it's too long
        max_chars_per_line = 40
        text_lines = [display_text[i:i+max_chars_per_line] 
                     for i in range(0, len(display_text), max_chars_per_line)]
        
        for i, line in enumerate(text_lines):
            cv2.putText(
                frame,
                line,
                (10, 40 + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Display word suggestions if available
        if self.word_suggestions:
            y_pos = 200
            cv2.putText(
                frame,
                "Suggestions:",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            for i, suggestion in enumerate(self.word_suggestions, 1):
                y_pos += 30
                cv2.putText(
                    frame,
                    f"{i}. {suggestion}",
                    (30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Recognize gesture
                gesture = self.recognize_gesture(hand_landmarks)
                
                # Add gesture to text with delay
                self.add_gesture_to_text(gesture)
                
                # Display gesture with confidence visualization
                confidence_color = (0, 255, 0) if gesture != "Unknown" else (0, 165, 255)
                cv2.putText(
                    frame,
                    f"Current Gesture: {gesture}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    confidence_color,
                    2
                )
                
                # Display timer if gesture is being held
                if self.current_gesture != "Unknown" and not self.gesture_confirmed:
                    time_held = time.time() - self.last_gesture_time
                    if time_held < self.gesture_delay:
                        progress = int((time_held / self.gesture_delay) * 100)
                        cv2.putText(
                            frame,
                            f"Hold: {progress}%",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2
                        )
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time)
        self.prev_frame_time = current_time
        self.fps_buffer.append(fps)
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        cv2.putText(
            frame,
            f"FPS: {int(avg_fps)}",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def run(self):
        print("\nStarting Custom Hand Gesture Recognition")
        print("Available gestures:", list(self.gesture_data.keys()))
        print("Press 'q' or ESC to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process frame
                try:
                    processed_frame = self.process_frame(frame)
                except Exception as e:
                    print(f"Frame processing error: {str(e)}")
                    processed_frame = frame
                
                # Display the frame
                cv2.imshow('Custom Hand Gesture Recognition', processed_frame)
                
                # Break loop with 'q' or ESC
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                
        except Exception as e:
            print(f"Runtime error: {str(e)}")
        finally:
            print("\nCleaning up resources...")
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    try:
        recognizer = CustomHandGestureRecognizer()
        recognizer.run()
    except FileNotFoundError:
        print("\nPlease run gesture_trainer.py first to create your custom gesture dataset!")
