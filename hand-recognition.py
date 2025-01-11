import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

class HandGestureRecognizer:
    def __init__(self):
        # Initialize video capture with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Initialize MediaPipe Hands with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,  # 0=Faster, 1=Balanced
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # Performance monitoring
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps_buffer = deque(maxlen=30)  # Store last 30 FPS values
        
        # Gesture smoothing
        self.gesture_buffer = deque(maxlen=5)  # Store last 5 gestures
        self.prev_landmarks = None
        
    def calculate_finger_angles(self, landmarks, frame_shape):
        # Convert landmarks to numpy array with pixel coordinates
        image_height, image_width = frame_shape[:2]
        points = np.array([[int(lm.x * image_width), int(lm.y * image_height)] for lm in landmarks.landmark])
        
        # Define finger joints
        finger_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        finger_states = []
        for finger, indices in finger_indices.items():
            # Get joint coordinates
            base = points[indices[0]]
            mid = points[indices[1]]
            tip = points[indices[3]]
            
            # Calculate vectors
            v1 = mid - base
            v2 = tip - mid
            
            # Calculate angle
            angle = np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))
            
            # Dynamic thresholds based on finger type
            if finger == 'thumb':
                threshold = 140
            elif finger in ['index', 'middle']:
                threshold = 165
            else:
                threshold = 155
                
            finger_states.append(abs(angle) > threshold)
            
        return finger_states
    
    def smooth_landmarks(self, current_landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
        
        alpha = 0.7  # Smoothing factor
        smoothed_landmarks = []
        for curr, prev in zip(current_landmarks.landmark, self.prev_landmarks.landmark):
            smoothed_x = alpha * prev.x + (1 - alpha) * curr.x
            smoothed_y = alpha * prev.y + (1 - alpha) * curr.y
            smoothed_z = alpha * prev.z + (1 - alpha) * curr.z
            
            curr.x, curr.y, curr.z = smoothed_x, smoothed_y, smoothed_z
            
        self.prev_landmarks = current_landmarks
        return current_landmarks
    
    def recognize_gesture(self, finger_states):
        # Enhanced gesture mapping with more precise definitions
        gesture_map = {
            (0, 0, 0, 0, 0): 'A',  # Fist
            (0, 1, 0, 0, 0): 'B',  # Index pointing
            (0, 1, 1, 0, 0): 'V',  # Victory
            (0, 1, 1, 1, 0): 'W',  # Three fingers
            (1, 1, 1, 1, 1): 'O',  # Open palm
            (1, 0, 0, 0, 1): 'Y',  # Rock on
            (1, 1, 0, 0, 0): 'L',  # L shape
            (0, 0, 0, 0, 1): 'I'   # Pinky only
        }
        
        current_gesture = gesture_map.get(tuple(finger_states), 'Unknown')
        self.gesture_buffer.append(current_gesture)
        
        # Return most common gesture in buffer for stability
        if len(self.gesture_buffer) >= 3:
            from collections import Counter
            return Counter(self.gesture_buffer).most_common(1)[0][0]
        return current_gesture
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Apply landmark smoothing
                smoothed_landmarks = self.smooth_landmarks(hand_landmarks)
                
                # Enhanced visualization
                self.mp_draw.draw_landmarks(
                    frame,
                    smoothed_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )
                
                # Get finger states and recognize gesture
                finger_states = self.calculate_finger_angles(smoothed_landmarks, frame.shape)
                gesture = self.recognize_gesture(finger_states)
                
                # Display gesture with confidence visualization
                confidence_color = (0, 255, 0) if gesture != 'Unknown' else (0, 165, 255)
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    confidence_color,
                    2
                )
        
        # Calculate and smooth FPS
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.curr_frame_time
        self.fps_buffer.append(fps)
        
        # Display smoothed FPS
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
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame - check camera connection")
                    break
                
                # Flip frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process frame with error handling
                try:
                    processed_frame = self.process_frame(frame)
                except Exception as e:
                    print(f"Frame processing error: {str(e)}")
                    processed_frame = frame  # Fallback to original frame
                
                # Display the frame
                cv2.imshow('Hand Gesture Recognition', processed_frame)
                
                # Break loop with 'q' or ESC
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # 27 is ESC key
                    break
                
        except Exception as e:
            print(f"Runtime error: {str(e)}")
        finally:
            print("Cleaning up resources...")
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    recognizer = HandGestureRecognizer()
    recognizer.run()