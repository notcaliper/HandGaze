import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import pickle
import os
from typing import List, Dict, Optional
from offline_dictionary import OfflineDictionary

class CustomHandGestureRecognizer:
    def __init__(self):
        # Initialize video capture with optimized settings for Linux
        self.cap = cv2.VideoCapture(0)
        # Set V4L2 (Video4Linux2) settings for better Linux performance
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        
        # Verify camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera. Please check camera permissions and connections.")
        
        # Initialize MediaPipe Hands with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,  # Increased from 0.5 to 0.6 for more reliable detection
            min_tracking_confidence=0.6    # Increased from 0.5 to 0.6 for more reliable tracking
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
        self.fps_buffer = deque(maxlen=10)
        self.prev_frame_time = 0
        
        # Gesture smoothing with increased buffer for more stability
        self.gesture_buffer = deque(maxlen=5)  # Increased from 3 to 5 for better stability
        
        # Initialize offline dictionary
        self.dictionary_helper = OfflineDictionary()
        
        # Add word building variables
        self.current_word = ""
        self.word_suggestions: List[str] = []
        self.show_suggestions = False
        
        # Add gesture typing variables
        self.typed_text = ""
        self.last_gesture = "Unknown"
        self.current_gesture = "Unknown"
        self.sentence = ""  # Add sentence variable to store complete text
        
        # New cooldown-based input system
        self.input_cooldown = 1.5  # Cooldown period in seconds - Increased to 1.5 for more obvious cooldown
        self.next_input_time = 0   # Timestamp when next input will be accepted
        self.gesture_hold_time = 0.7  # How long to hold a gesture to confirm it - Increased to make it easier to see
        self.gesture_hold_start = 0    # When the current gesture started being held
        self.system_state = "READY"  # Track system state explicitly: "READY", "HOLD", or "COOLDOWN"
        self.last_processed_gesture = None  # Track the last successfully processed gesture
        self.debug_info = ""  # For displaying debug information
        
        # Error recovery
        self.last_safety_check = time.time()
        self.safety_check_interval = 0.5  # Check for errors every 0.5 seconds
        self.error_count = 0  # Keep track of how many errors have occurred
        
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
        """Calculate relative distances between key points with error protection"""
        try:
            # Validate input
            if landmarks is None:
                return np.zeros(10, dtype=np.float32)  # Return zeros for safety
                
            key_points = [4, 8, 12, 16, 20]
            n_distances = (len(key_points) * (len(key_points) - 1)) // 2
            distances = np.zeros(n_distances, dtype=np.float32)
            
            idx = 0
            for i in range(len(key_points)):
                # Check index bounds
                if key_points[i] >= len(landmarks):
                    continue
                    
                p1 = np.array(landmarks[key_points[i]], dtype=np.float32)
                for j in range(i + 1, len(key_points)):
                    # Check index bounds
                    if key_points[j] >= len(landmarks):
                        continue
                        
                    p2 = np.array(landmarks[key_points[j]], dtype=np.float32)
                    dist = np.sqrt(np.sum((p1 - p2) ** 2))
                    
                    # Avoid out of bounds index
                    if idx < n_distances:
                        distances[idx] = dist
                        idx += 1
                    
            return distances
            
        except Exception as e:
            print(f"Error in relative distances calculation: {str(e)}")
            return np.zeros(10, dtype=np.float32)  # Return zeros for safety

    def calculate_angles(self, landmarks):
        """Calculate angles between finger joints with error protection"""
        try:
            # Validate input
            if landmarks is None:
                return [0.0] * 15  # Return zeros for safety
                
            angles = []
            # Define finger joint connections (indices)
            finger_joints = [
                [0, 1, 2], [1, 2, 3], [2, 3, 4],  # Thumb
                [0, 5, 6], [5, 6, 7], [6, 7, 8],  # Index
                [0, 9, 10], [9, 10, 11], [10, 11, 12],  # Middle
                [0, 13, 14], [13, 14, 15], [14, 15, 16],  # Ring
                [0, 17, 18], [17, 18, 19], [18, 19, 20]  # Pinky
            ]
            
            for p1, p2, p3 in finger_joints:
                # Check index bounds
                if max(p1, p2, p3) >= len(landmarks):
                    angles.append(0.0)
                    continue
                    
                v1 = np.array([
                    landmarks[p1][0] - landmarks[p2][0],
                    landmarks[p1][1] - landmarks[p2][1]
                ], dtype=np.float32)
                
                v2 = np.array([
                    landmarks[p3][0] - landmarks[p2][0],
                    landmarks[p3][1] - landmarks[p2][1]
                ], dtype=np.float32)
                
                # Normalize vectors
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm == 0.0 or v2_norm == 0.0:
                    angles.append(0.0)
                    continue
                
                v1_normalized = v1 / v1_norm
                v2_normalized = v2 / v2_norm
                
                # Calculate angle
                dot_product = np.dot(v1_normalized, v2_normalized)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angles.append(float(angle))
                
            # Ensure we return the expected number of angles
            if len(angles) < 15:
                angles.extend([0.0] * (15 - len(angles)))
                
            return angles
            
        except Exception as e:
            print(f"Error in angle calculation: {str(e)}")
            return [0.0] * 15  # Return zeros for safety

    def calculate_landmark_similarity(self, landmarks1, landmarks2_features):
        """Calculate similarity between landmarks with error protection"""
        try:
            # Verify inputs are valid
            if landmarks1 is None or landmarks2_features is None:
                return float('inf')  # Return worst possible similarity score
                
            if 'angles' not in landmarks2_features or 'rel_distances' not in landmarks2_features:
                return float('inf')
                
            # Calculate angles and relative distances for the current landmarks
            angles1 = np.array(self.calculate_angles(landmarks1), dtype=np.float32)
            rel_dist1 = np.array(self.get_relative_distances(landmarks1), dtype=np.float32)
            
            # Get the pre-computed features for comparison
            angles2 = np.array(landmarks2_features['angles'], dtype=np.float32)
            rel_dist2 = np.array(landmarks2_features['rel_distances'], dtype=np.float32)
            
            # Check array sizes match
            if angles1.size != angles2.size or rel_dist1.size != rel_dist2.size:
                return float('inf')
            
            # Calculate differences
            angle_diff = np.mean(np.abs(angles1 - angles2))
            rel_dist_diff = np.mean(np.abs(rel_dist1 - rel_dist2))
            
            # Combine similarities with weights
            return float(angle_diff * 0.6 + rel_dist_diff * 0.4)
            
        except Exception as e:
            print(f"Error in landmark similarity calculation: {str(e)}")
            return float('inf')  # Return worst possible similarity score

    def recognize_gesture(self, current_landmarks):
        """Optimized gesture recognition with stability improvements and error handling"""
        try:
            if not current_landmarks:
                return "Unknown"
            
            # Safely convert landmarks to list, with error handling
            try:
                current_landmarks_list = [[lm.x, lm.y] for lm in current_landmarks.landmark]
                if not current_landmarks_list:
                    return "Unknown"
            except (AttributeError, TypeError):
                # Handle case where landmarks or landmark.attribute doesn't exist
                return "Unknown"
            
            # Compare with pre-computed features
            best_score = float('inf')
            best_gesture = "Unknown"
            second_best_score = float('inf')
            
            for gesture_name, gesture_features in self.precomputed_gesture_features.items():
                # Check if gesture_features is valid
                if not gesture_features:
                    continue
                    
                # Compare with multiple samples for each gesture for better accuracy
                try:
                    scores = [self.calculate_landmark_similarity(current_landmarks_list, features) 
                            for features in gesture_features[:5] if features]  # Use more samples (up to 5)
                    
                    # Skip if no valid scores
                    if not scores:
                        continue
                    
                    # Use the best match from each gesture's samples
                    min_score = min(scores) if scores else float('inf')
                    
                    if min_score < best_score:
                        second_best_score = best_score
                        best_score = min_score
                        best_gesture = gesture_name
                except Exception as e:
                    print(f"Error comparing with gesture {gesture_name}: {str(e)}")
                    continue
            
            # Calculate confidence as the difference between best and second best
            confidence = second_best_score - best_score
            
            # Apply stricter threshold checks for better recognition
            if best_score > 0.20:  # Overall threshold
                best_gesture = "Unknown"
            elif confidence < 0.05 and best_score > 0.15:  # Ambiguous gestures
                best_gesture = "Unknown"
            
            # Stability buffer to prevent flickering
            try:
                if best_gesture != self.last_gesture:
                    # Only accept a new gesture if it's recognized with high confidence
                    if best_gesture == "Unknown" or best_score < 0.12:
                        self.gesture_buffer.append(best_gesture)
                        
                        # Only change the gesture if it appears consistently
                        if len(set(self.gesture_buffer)) == 1:
                            self.last_gesture = best_gesture
                else:
                    # If the same gesture is recognized again, reinforce it
                    self.gesture_buffer.append(best_gesture)
            except Exception as e:
                print(f"Error in gesture buffer handling: {str(e)}")
            
            return self.last_gesture
            
        except Exception as e:
            print(f"Error in gesture recognition: {str(e)}")
            return "Unknown"

    def add_gesture_to_text(self, gesture: str) -> None:
        """
        Add recognized gesture to typed text with absolute state enforcement to prevent glitches
        """
        try:
            current_time = time.time()
            self.debug_info = f"State: {self.system_state}, Gesture: {gesture}"
            
            # STATE: COOLDOWN (System is in mandatory waiting period after processing a gesture)
            if self.system_state == "COOLDOWN":
                # Stay in cooldown until the timer expires
                if current_time >= self.next_input_time:
                    self.system_state = "READY"
                    self.debug_info += " | Cooldown complete, now READY"
                else:
                    # Still in cooldown - ignore all gestures and inputs
                    self.debug_info += f" | Cooldown remaining: {self.next_input_time - current_time:.1f}s"
                    return
            
            # STATE: HOLD (User is holding a gesture to confirm it)
            elif self.system_state == "HOLD":
                # If the gesture changed during hold, reset to holding the new gesture
                if gesture != self.current_gesture and gesture != "Unknown":
                    self.current_gesture = gesture
                    self.gesture_hold_start = current_time
                    self.debug_info += " | Hold reset - gesture changed"
                    return
                    
                # If the gesture became Unknown, cancel the hold
                if gesture == "Unknown":
                    self.system_state = "READY"
                    self.current_gesture = "Unknown"
                    self.debug_info += " | Hold canceled - gesture lost"
                    return
                    
                # Check if we've held the gesture long enough to process it
                if (current_time - self.gesture_hold_start) >= self.gesture_hold_time:
                    # GESTURE CONFIRMED - Process it now
                    self.debug_info += " | Gesture confirmed - processing"
                    
                    # Record which gesture was processed
                    self.last_processed_gesture = gesture
                    
                    # Process the gesture input
                    try:
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
                    except Exception as e:
                        print(f"Error processing gesture input: {str(e)}")
                    
                    # ALWAYS transition to COOLDOWN state after processing a gesture
                    # This is the most important part - ensures cooldown happens even if processing fails
                    self.system_state = "COOLDOWN"
                    self.next_input_time = current_time + self.input_cooldown
                    self.debug_info += f" | Now in COOLDOWN for {self.input_cooldown}s"
                    
                    print(f"Gesture processed: {gesture}")
                    print(f"Current word: {self.current_word}")
                    print(f"Suggestions: {self.word_suggestions}")
                else:
                    # Still holding - continue the hold
                    self.debug_info += f" | Holding {(current_time - self.gesture_hold_start):.1f}s/{self.gesture_hold_time}s"
            
            # STATE: READY (System is ready to accept a new gesture)
            elif self.system_state == "READY":
                # Ignore Unknown gestures
                if gesture == "Unknown":
                    return
                    
                # New gesture detected - start the hold process
                self.system_state = "HOLD"
                self.current_gesture = gesture
                self.gesture_hold_start = current_time
                self.debug_info += f" | Started holding {gesture}"
                
        except Exception as e:
            print(f"Critical error in gesture processing: {str(e)}")
            # Safety fallback - if any error occurs during processing, revert to READY state
            # but only if we're not in cooldown or cooldown has expired
            current_time = time.time()
            if self.system_state != "COOLDOWN" or current_time >= self.next_input_time:
                self.system_state = "READY"
                self.debug_info = f"ERROR: {str(e)[:30]}"

    def process_frame(self, frame):
        try:
            # Run system health check to catch and fix errors
            self.check_system_health()
            
            # Convert BGR to RGB and ensure array is contiguous
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Create a semi-transparent overlay
            overlay = frame.copy()
            
            # Add a dark semi-transparent background for better text visibility
            cv2.rectangle(overlay, (0, 0), (400, 250), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Display complete sentence and current word with better formatting
            display_text = self.sentence + self.current_word
            max_chars_per_line = 40
            text_lines = [display_text[i:i+max_chars_per_line] 
                        for i in range(0, len(display_text), max_chars_per_line)]
            
            # Add text background
            text_y = 40
            for line in text_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (8, text_y - 25), (text_size[0] + 15, text_y + 5), (50, 50, 50), -1)
                cv2.putText(
                    frame,
                    line,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                text_y += 30
            
            # Display word suggestions with better styling
            if self.word_suggestions:
                y_pos = 200
                # Add suggestion header
                cv2.putText(
                    frame,
                    "Suggestions:",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 223, 0),  # Golden yellow
                    2
                )
                # Add suggestions with highlight for the best match
                for i, suggestion in enumerate(self.word_suggestions, 1):
                    y_pos += 30
                    # Background for suggestion
                    text_size = cv2.getTextSize(f"{i}. {suggestion}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (28, y_pos - 25), (text_size[0] + 35, y_pos + 5), 
                                (40, 40, 40) if i > 1 else (0, 100, 0), -1)
                    cv2.putText(
                        frame,
                        f"{i}. {suggestion}",
                        (30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0) if i > 1 else (255, 255, 255),
                        2
                    )
            
            # Get current time for cooldown display
            current_time = time.time()
            
            # Process hand landmarks if detected
            current_gesture = "Unknown"
            try:
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks with custom styling
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)
                        )
                        
                        # Recognize gesture
                        current_gesture = self.recognize_gesture(hand_landmarks)
                        
                        # Display gesture with modern styling
                        confidence_color = (0, 255, 150) if current_gesture != "Unknown" else (0, 165, 255)
                        cv2.rectangle(frame, (8, 55), (300, 85), (50, 50, 50), -1)
                        cv2.putText(
                            frame,
                            f"Gesture: {current_gesture}",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            confidence_color,
                            2
                        )
            except Exception as e:
                print(f"Error processing hand landmarks: {str(e)}")
                # Make sure we have a valid gesture if there was an error
                current_gesture = "Unknown"
            
            # Always process the current gesture, even if no hand is detected (Unknown)
            try:
                self.add_gesture_to_text(current_gesture)
            except Exception as e:
                print(f"Error in add_gesture_to_text: {str(e)}")
                # If the gesture processing fails, ensure we don't get stuck
                if current_time >= self.next_input_time:
                    self.system_state = "READY"
            
            # Display system state (Always visible, even when no hand is detected)
            # COOLDOWN state display
            try:
                if self.system_state == "COOLDOWN":
                    # Calculate cooldown progress
                    cooldown_remaining = max(0, self.next_input_time - current_time)
                    cooldown_progress = int(((self.input_cooldown - cooldown_remaining) / self.input_cooldown) * 100)
                    
                    # Background for cooldown bar
                    cv2.rectangle(frame, (10, 100), (210, 120), (50, 50, 50), -1)
                    
                    # Cooldown progress bar
                    bar_width = int(200 * (cooldown_progress / 100))
                    cv2.rectangle(frame, (10, 100), (10 + bar_width, 120), (0, 100, 255), -1)
                    
                    # Cooldown text
                    cv2.putText(
                        frame,
                        f"Cooldown: {cooldown_remaining:.1f}s",
                        (220, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 100, 255),
                        2
                    )
                    
                    # "WAIT" status
                    cv2.rectangle(frame, (8, 130), (170, 160), (0, 0, 140), -1)
                    cv2.putText(
                        frame,
                        "WAIT",
                        (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                    
                    # Show which gesture was just processed
                    if self.last_processed_gesture:
                        cv2.rectangle(frame, (180, 130), (380, 160), (60, 60, 60), -1)
                        cv2.putText(
                            frame,
                            f"Last: {self.last_processed_gesture}",
                            (190, 155),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 200, 0),
                            2
                        )
                
                # HOLD state display
                elif self.system_state == "HOLD":
                    # Calculate hold progress
                    time_held = current_time - self.gesture_hold_start
                    hold_progress = int((time_held / self.gesture_hold_time) * 100)
                    
                    # Background for hold bar
                    cv2.rectangle(frame, (10, 100), (210, 120), (50, 50, 50), -1)
                    
                    # Hold progress bar
                    bar_width = int(200 * (hold_progress / 100))
                    cv2.rectangle(frame, (10, 100), (10 + bar_width, 120), (0, 255, 0), -1)
                    
                    # Hold text
                    cv2.putText(
                        frame,
                        f"Hold: {hold_progress}%",
                        (220, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    
                    # "HOLD" status
                    cv2.rectangle(frame, (8, 130), (170, 160), (0, 140, 70), -1)
                    cv2.putText(
                        frame,
                        "HOLD",
                        (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                    
                    # Show which gesture is being held
                    cv2.rectangle(frame, (180, 130), (380, 160), (60, 60, 60), -1)
                    cv2.putText(
                        frame,
                        f"Holding: {self.current_gesture}",
                        (190, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 200),
                        2
                    )
                
                # READY state display
                else:  # self.system_state == "READY"
                    # System is ready for input - show ready status
                    cv2.rectangle(frame, (8, 130), (170, 160), (0, 140, 0), -1)
                    cv2.putText(
                        frame,
                        "READY",
                        (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
            except Exception as e:
                # If display fails, at least show an error message
                print(f"Error displaying system state: {str(e)}")
                cv2.rectangle(frame, (8, 130), (300, 160), (255, 0, 0), -1)
                cv2.putText(
                    frame,
                    "ERROR: " + str(e)[:15],
                    (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Add debug information
            cv2.rectangle(frame, (8, 380), (630, 410), (40, 40, 40), -1)
            cv2.putText(
                frame,
                self.debug_info,
                (10, 405),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
            
            # Calculate and display FPS with styling
            current_time = time.time()
            fps = 1 / (current_time - self.prev_frame_time)
            self.prev_frame_time = current_time
            self.fps_buffer.append(fps)
            avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
            
            # FPS counter with background
            cv2.rectangle(frame, (8, 170), (120, 200), (50, 50, 50), -1)
            cv2.putText(
                frame,
                f"FPS: {int(avg_fps)}",
                (10, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Add safety timeout for states
            # If the system has been in HOLD state too long, reset it
            if self.system_state == "HOLD" and (current_time - self.gesture_hold_start) > self.gesture_hold_time * 3:
                print("Safety timeout: HOLD state lasted too long")
                self.system_state = "READY"
            
            # If cooldown has expired but state wasn't updated, fix it
            if self.system_state == "COOLDOWN" and current_time >= self.next_input_time:
                print("Safety check: Cooldown expired but state wasn't updated")
                self.system_state = "READY"
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            # Create a fallback frame with error message in case of crash
            try:
                # Display error message
                cv2.rectangle(frame, (10, 200), (630, 280), (0, 0, 200), -1)
                cv2.putText(
                    frame,
                    "ERROR: " + str(e),
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                
                # Safety reset - ensure system state is reset if error occurs
                if current_time >= self.next_input_time:
                    self.system_state = "READY"
                
                return frame
            except:
                # Last resort - return original frame if error display fails
                return frame
    
    def run(self):
        print("\nStarting Custom Hand Gesture Recognition")
        print("Available gestures:", list(self.gesture_data.keys()))
        print("Press 'q' or ESC to quit")
        
        # Keep track of consecutive frame errors
        consecutive_errors = 0
        max_consecutive_errors = 5
        crash_recovery_time = time.time()
        
        try:
            while True:
                try:
                    # Run health check regardless of frame status
                    self.check_system_health()
                    
                    # Capture frame
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        time.sleep(0.1)  # Short delay to avoid tight loop
                        continue
                    
                    # Flip frame for natural interaction
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame with comprehensive error handling
                    try:
                        processed_frame = self.process_frame(frame)
                        
                        # If we successfully processed a frame, reset error counter
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        print(f"Frame processing error: {str(e)}")
                        
                        # Create error message on frame
                        cv2.rectangle(frame, (10, 200), (630, 280), (0, 0, 200), -1)
                        cv2.putText(
                            frame,
                            f"ERROR: {str(e)[:60]}",
                            (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2
                        )
                        
                        # Add current system state to help debug
                        cv2.putText(
                            frame,
                            f"State: {self.system_state} | Errors: {consecutive_errors}",
                            (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 200, 0),
                            2
                        )
                        
                        # Use original frame as fallback
                        processed_frame = frame
                        
                    # Emergency recovery if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        if time.time() - crash_recovery_time > 3.0:  # Only reset every 3 seconds
                            print("\n" + "!"*50)
                            print("CRITICAL ERROR: System experiencing multiple failures")
                            print("Performing emergency recovery...")
                            print("!"*50 + "\n")
                            
                            # Force system back to READY state
                            self.system_state = "READY"
                            self.current_gesture = "Unknown"
                            self.gesture_hold_start = 0
                            self.next_input_time = 0
                            self.error_count = 0
                            consecutive_errors = 0
                            crash_recovery_time = time.time()
                            
                            # Add critical error overlay
                            cv2.rectangle(processed_frame, (0, 0), (640, 480), (0, 0, 200), 20)
                            cv2.rectangle(processed_frame, (50, 180), (590, 300), (0, 0, 150), -1)
                            cv2.putText(
                                processed_frame,
                                "EMERGENCY RECOVERY",
                                (100, 220),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (255, 255, 255),
                                3
                            )
                            cv2.putText(
                                processed_frame,
                                "System has been reset",
                                (150, 270),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 0),
                                2
                            )
                    
                    # Always display the frame, even if processing failed
                    cv2.imshow('Custom Hand Gesture Recognition', processed_frame)
                    
                    # Break loop with 'q' or ESC
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), 27]:
                        break
                        
                except Exception as e:
                    # Catch-all for any error in the main loop
                    consecutive_errors += 1
                    print(f"Runtime loop error: {str(e)}")
                    time.sleep(0.1)  # Prevent tight loop in case of persistent errors
                    
                    # Force state reset if main loop has critical error
                    if consecutive_errors >= max_consecutive_errors:
                        print("Critical main loop error - forcing state reset")
                        self.system_state = "READY"
                        self.error_count = 0
                
        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
        finally:
            print("\nCleaning up resources...")
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

    def check_system_health(self):
        """
        Performs safety checks and corrections to prevent the system from getting stuck
        """
        try:
            current_time = time.time()
            
            # Only run checks every safety_check_interval seconds
            if current_time - self.last_safety_check < self.safety_check_interval:
                return
                
            self.last_safety_check = current_time
            
            # Check 1: If cooldown has expired but we're still in COOLDOWN state
            if self.system_state == "COOLDOWN" and current_time >= self.next_input_time:
                print("SAFETY: Cooldown expired but state wasn't updated - fixing")
                self.system_state = "READY"
                self.current_gesture = "Unknown"
            
            # Check 2: If we've been in HOLD state too long (3x the hold time)
            if self.system_state == "HOLD" and (current_time - self.gesture_hold_start) > self.gesture_hold_time * 3:
                print("SAFETY: Hold state lasted too long - resetting")
                self.system_state = "READY"
                self.current_gesture = "Unknown"
            
            # Check 3: If gesture_hold_start is in the future (clock got adjusted)
            if self.gesture_hold_start > current_time:
                print("SAFETY: Hold start time is in the future - fixing")
                self.gesture_hold_start = current_time - self.gesture_hold_time
            
            # Check 4: If next_input_time is in the future but by too much
            if self.next_input_time > current_time + self.input_cooldown * 2:
                print("SAFETY: Cooldown end time is too far in the future - fixing")
                self.next_input_time = current_time + self.input_cooldown
            
        except Exception as e:
            print(f"Error in health check: {str(e)}")
            # Last resort recovery
            self.error_count += 1
            if self.error_count > 5:
                print("CRITICAL: Multiple errors detected - performing emergency reset")
                self.system_state = "READY"
                self.current_gesture = "Unknown"
                self.gesture_hold_start = 0
                self.next_input_time = 0
                self.error_count = 0

if __name__ == "__main__":
    try:
        recognizer = CustomHandGestureRecognizer()
        recognizer.run()
    except FileNotFoundError:
        print("\nPlease run gesture_trainer.py first to create your custom gesture dataset!")
