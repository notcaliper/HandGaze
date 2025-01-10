import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import pickle
import os

class CustomHandGestureRecognizer:
    def __init__(self):
        # Initialize video capture with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load gesture data
        self.gesture_data = self.load_latest_gesture_data()
        if not self.gesture_data:
            print("No gesture data found! Please run gesture_trainer.py first.")
            raise FileNotFoundError("No gesture data available")
        
        # Performance monitoring
        self.fps_buffer = deque(maxlen=30)
        self.prev_frame_time = 0
        
        # Gesture smoothing
        self.gesture_buffer = deque(maxlen=5)
    
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
    
    def calculate_angles(self, landmarks):
        """Calculate angles between finger joints for better recognition"""
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
            v1 = np.array([landmarks[p1][0] - landmarks[p2][0], 
                          landmarks[p1][1] - landmarks[p2][1]])
            v2 = np.array([landmarks[p3][0] - landmarks[p2][0], 
                          landmarks[p3][1] - landmarks[p2][1]])
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm == 0 or v2_norm == 0:
                angles.append(0)
                continue
                
            v1_normalized = v1 / v1_norm
            v2_normalized = v2 / v2_norm
            
            # Calculate angle
            angle = np.arccos(np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0))
            angles.append(angle)
            
        return angles

    def calculate_landmark_similarity(self, landmarks1, landmarks2):
        """Calculate similarity between two sets of landmarks using multiple features"""
        # Convert landmarks to numpy arrays
        landmarks1_array = np.array([[p[0], p[1]] for p in landmarks1])
        landmarks2_array = np.array([[p[0], p[1]] for p in landmarks2])
        
        # 1. Position similarity (weighted less now)
        position_distances = np.sqrt(np.sum((landmarks1_array - landmarks2_array) ** 2, axis=1))
        position_similarity = np.mean(position_distances) * 0.4  # Reduced weight
        
        # 2. Angle similarity (weighted more)
        angles1 = self.calculate_angles(landmarks1)
        angles2 = self.calculate_angles(landmarks2)
        angle_diff = np.mean(np.abs(np.array(angles1) - np.array(angles2)))
        angle_similarity = angle_diff * 0.6  # Increased weight
        
        # 3. Calculate relative distances between key points
        def get_relative_distances(landmarks):
            # Key points: thumb tip, index tip, middle tip, ring tip, pinky tip
            key_points = [4, 8, 12, 16, 20]
            distances = []
            for i in range(len(key_points)):
                for j in range(i + 1, len(key_points)):
                    p1 = landmarks[key_points[i]]
                    p2 = landmarks[key_points[j]]
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    distances.append(dist)
            return np.array(distances)
        
        rel_dist1 = get_relative_distances(landmarks1)
        rel_dist2 = get_relative_distances(landmarks2)
        relative_similarity = np.mean(np.abs(rel_dist1 - rel_dist2))
        
        # Combine similarities with weights
        total_similarity = (position_similarity + angle_similarity + relative_similarity) / 3
        return total_similarity
        
    def recognize_gesture(self, current_landmarks):
        """Recognize gesture by comparing with trained data"""
        if not current_landmarks:
            return "Unknown"
            
        # Convert current landmarks to list format
        current_landmarks_list = [[lm.x, lm.y] for lm in current_landmarks.landmark]
        
        # Compare with each stored gesture
        gesture_scores = {}
        
        for gesture_name, gesture_samples in self.gesture_data.items():
            similarities = []
            for sample in gesture_samples:
                sample_2d = [[p[0], p[1]] for p in sample]
                similarity = self.calculate_landmark_similarity(current_landmarks_list, sample_2d)
                similarities.append(similarity)
            
            # Use the average of top 3 best matches for this gesture
            top_similarities = sorted(similarities)[:3]
            gesture_scores[gesture_name] = np.mean(top_similarities) if top_similarities else float('inf')
        
        # Find the best match
        if not gesture_scores:
            return "Unknown"
            
        best_gesture = min(gesture_scores.items(), key=lambda x: x[1])
        
        # More stringent threshold
        if best_gesture[1] > 0.25:  # Adjusted threshold
            return "Unknown"
            
        # Enhanced gesture smoothing with confidence weighting
        self.gesture_buffer.append((best_gesture[0], best_gesture[1]))
        
        if len(self.gesture_buffer) >= 3:
            # Weight recent gestures more heavily
            weights = np.exp(-0.5 * np.arange(len(self.gesture_buffer)))
            weights = weights / np.sum(weights)
            
            # Count weighted occurrences
            gesture_counts = {}
            for (gesture, score), weight in zip(self.gesture_buffer, weights):
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + weight
                
            return max(gesture_counts.items(), key=lambda x: x[1])[0]
            
        return best_gesture[0]
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
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
                
                # Display gesture with confidence visualization
                confidence_color = (0, 255, 0) if gesture != "Unknown" else (0, 165, 255)
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    confidence_color,
                    2
                )
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = current_time
        self.fps_buffer.append(fps)
        
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0
        cv2.putText(
            frame,
            f"FPS: {int(avg_fps)}",
            (10, 90),
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
