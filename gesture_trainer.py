import cv2
import numpy as np
import os
import mediapipe as mp
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class GestureTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create directory for storing gesture data
        self.data_dir = 'gesture_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Fixed filenames for gesture data
        self.data_file = os.path.join(self.data_dir, 'gesture_data.pkl')
        self.summary_file = os.path.join(self.data_dir, 'gesture_summary.txt')
        self.backup_dir = os.path.join(self.data_dir, 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Load existing data if available
        self.gesture_data = self.load_existing_data()
        
    def load_existing_data(self) -> Dict:
        """Load existing gesture data if available"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                # Migrate old data format to new format if needed
                migrated_data = {}
                for gesture_name, gesture_data in data.items():
                    if isinstance(gesture_data, list):  # Old format
                        migrated_data[gesture_name] = {
                            'landmarks': gesture_data,
                            'timestamp': datetime.now().isoformat(),
                            'samples': len(gesture_data)
                        }
                    else:  # New format
                        migrated_data[gesture_name] = gesture_data
                print(f"Loaded {len(migrated_data)} existing gestures")
                return migrated_data
            except Exception as e:
                print(f"Error loading data: {e}")
                self._create_backup(self.data_file)
        return {}
    
    def _create_backup(self, file_path: str) -> None:
        """Create a backup of a file with timestamp"""
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}.pkl")
            try:
                os.rename(file_path, backup_path)
                print(f"Created backup at {backup_path}")
            except Exception as e:
                print(f"Failed to create backup: {e}")
    
    def _preprocess_landmarks(self, landmarks_list: List) -> List:
        """Preprocess landmarks to normalize and reduce noise"""
        if not landmarks_list:
            return []
            
        # Convert to numpy array for processing
        landmarks = np.array(landmarks_list)
        
        # Normalize coordinates to be relative to hand center
        center = np.mean(landmarks, axis=0)
        normalized = landmarks - center
        
        # Scale to unit size
        scale = np.max(np.abs(normalized))
        if scale > 0:
            normalized /= scale
            
        return normalized.tolist()
    
    def _validate_hand_position(self, results) -> Tuple[bool, str]:
        """Validate hand position and provide feedback"""
        if not results.multi_hand_landmarks:
            return False, "No hand detected"
            
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Check if hand is too close to frame edges
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        if min(x_coords) < 0.1 or max(x_coords) > 0.9:
            return False, "Move hand away from left/right edges"
        if min(y_coords) < 0.1 or max(y_coords) > 0.9:
            return False, "Move hand away from top/bottom edges"
            
        return True, "Hand position good"
    
    def capture_gesture(self, gesture_name: str, num_samples: int = 20) -> bool:
        """Capture and save hand landmarks for a gesture"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        samples_collected = 0
        landmarks_data = []
        
        print(f"\nCollecting samples for gesture '{gesture_name}'")
        print("Press 'SPACE' to capture a sample, 'r' to redo last sample, 'q' to finish")
        
        last_sample = None
        feedback_message = ""
        feedback_color = (0, 255, 0)
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                # Validate hand position
                valid, message = self._validate_hand_position(results)
                feedback_message = message
                feedback_color = (0, 255, 0) if valid else (0, 0, 255)
            
            # Display UI
            # Progress bar
            progress = int((samples_collected / num_samples) * 200)
            cv2.rectangle(frame, (20, 50), (220, 70), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 50), (20 + progress, 70), (0, 255, 0), -1)
            
            # Text overlays
            cv2.putText(frame, f"Samples: {samples_collected}/{num_samples}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Gesture: {gesture_name}",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback_message,
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
            
            cv2.imshow('Gesture Training', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r') and last_sample is not None:
                if samples_collected > 0:
                    landmarks_data.pop()
                    samples_collected -= 1
                    print("Removed last sample")
            elif key == ord(' '):  # Space key
                if results.multi_hand_landmarks and valid:
                    # Convert landmarks to list and preprocess
                    landmarks = results.multi_hand_landmarks[0]
                    landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
                    processed_landmarks = self._preprocess_landmarks(landmarks_list)
                    
                    landmarks_data.append(processed_landmarks)
                    last_sample = processed_landmarks
                    samples_collected += 1
                    print(f"Sample {samples_collected} captured!")
                else:
                    print(feedback_message)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if landmarks_data:
            self.gesture_data[gesture_name] = {
                'landmarks': landmarks_data,
                'timestamp': datetime.now().isoformat(),
                'samples': len(landmarks_data)
            }
            return True
        return False
    
    def test_gesture(self, gesture_name: str) -> None:
        """Test a trained gesture with live feedback"""
        if gesture_name not in self.gesture_data:
            print(f"Gesture {gesture_name} not found!")
            return
            
        cap = cv2.VideoCapture(0)
        print("\nTesting gesture recognition. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Compare with trained gesture
                landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
                processed = self._preprocess_landmarks(landmarks_list)
                
                # Simple euclidean distance comparison
                trained_samples = self.gesture_data[gesture_name]['landmarks']
                distances = [np.mean(np.square(np.array(processed) - np.array(sample))) 
                           for sample in trained_samples]
                min_distance = min(distances)
                
                # Display match confidence
                confidence = max(0, 1 - min_distance)
                color = (0, int(255 * confidence), 0)
                cv2.putText(frame, f"Confidence: {confidence:.2f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Gesture Testing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_gesture_data(self) -> None:
        """Save collected gesture data to file"""
        # Create backup of existing file
        self._create_backup(self.data_file)
        
        # Save binary data
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.gesture_data, f)
        
        print(f"\nGesture data saved to {self.data_file}")
        
        # Save readable summary
        with open(self.summary_file, 'w') as f:
            f.write("Gesture Training Summary\n")
            f.write("=====================\n\n")
            for gesture, data in self.gesture_data.items():
                f.write(f"Gesture: {gesture}\n")
                f.write(f"Samples: {data['samples']}\n")
                f.write(f"Last updated: {data['timestamp']}\n")
                f.write("-----------------\n")
        
        print(f"Summary saved to {self.summary_file}")

def main():
    trainer = GestureTrainer()
    
    print("Welcome to Gesture Trainer!")
    print("This program will help you create your own gesture dataset.")
    
    # Show existing gestures if any
    if trainer.gesture_data:
        print("\nExisting gestures:", list(trainer.gesture_data.keys()))
    
    while True:
        print("\nOptions:")
        print("1. Add new gesture")
        print("2. Test existing gesture")
        print("3. Delete existing gesture")
        print("4. Save and exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            gesture_name = input("Enter gesture name (e.g., 'A', 'B', 'C'): ").upper()
            if gesture_name in trainer.gesture_data:
                overwrite = input("Gesture already exists. Overwrite? (y/n): ").lower()
                if overwrite != 'y':
                    continue
            
            num_samples = int(input("Enter number of samples to collect (recommended 20): "))
            
            print("\nGet ready to show your gesture!")
            print("Position your hand clearly in front of the camera.")
            input("Press Enter when ready...")
            
            if trainer.capture_gesture(gesture_name, num_samples):
                print(f"\nSuccessfully captured {gesture_name}!")
                trainer.save_gesture_data()  # Auto-save after capture
            else:
                print(f"\nFailed to capture {gesture_name}. Please try again.")
        
        elif choice == '2':
            if not trainer.gesture_data:
                print("\nNo gestures to test!")
                continue
                
            print("\nExisting gestures:", list(trainer.gesture_data.keys()))
            gesture_name = input("Enter gesture name to test: ").upper()
            if gesture_name in trainer.gesture_data:
                trainer.test_gesture(gesture_name)
            else:
                print("Gesture not found!")
        
        elif choice == '3':
            if not trainer.gesture_data:
                print("\nNo gestures to delete!")
                continue
                
            print("\nExisting gestures:", list(trainer.gesture_data.keys()))
            gesture_name = input("Enter gesture name to delete: ").upper()
            if gesture_name in trainer.gesture_data:
                confirm = input(f"Are you sure you want to delete {gesture_name}? (y/n): ").lower()
                if confirm == 'y':
                    del trainer.gesture_data[gesture_name]
                    print(f"Deleted gesture {gesture_name}")
                    trainer.save_gesture_data()  # Auto-save after deletion
            else:
                print("Gesture not found!")
        
        elif choice == '4':
            if trainer.gesture_data:
                trainer.save_gesture_data()
            print("\nThank you for using Gesture Trainer!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
