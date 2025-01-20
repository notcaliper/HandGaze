import cv2
import numpy as np
import os
import mediapipe as mp
import pickle
from typing import List, Dict
from datetime import datetime

class GestureTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # Create directory for storing gesture data
        self.data_dir = 'gesture_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create directory for gesture images
        self.image_dir = os.path.join(self.data_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        
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
                    return pickle.load(f)
            except:
                print("Could not load existing data, starting fresh")
                self._backup_data_file()
        return {}
    
    def _backup_data_file(self):
        """Create a backup of the data file if it exists"""
        if os.path.exists(self.data_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.backup_dir, f'gesture_data_backup_{timestamp}.pkl')
            try:
                with open(self.data_file, 'rb') as src, open(backup_file, 'wb') as dst:
                    dst.write(src.read())
                print(f"Backup created: {backup_file}")
            except Exception as e:
                print(f"Backup failed: {str(e)}")
    
    def _save_gesture_image(self, frame: np.ndarray, gesture_name: str, sample_num: int):
        """Save a snapshot of the gesture"""
        filename = os.path.join(self.image_dir, f'{gesture_name}_sample_{sample_num}.jpg')
        cv2.imwrite(filename, frame)
    
    def capture_gesture(self, gesture_name: str, num_samples: int = 20) -> bool:
        """Capture and save hand landmarks for a gesture"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        samples_collected = 0
        landmarks_data = []
        
        # Create progress window
        cv2.namedWindow('Gesture Training', cv2.WINDOW_NORMAL)
        
        print(f"\nCollecting samples for gesture '{gesture_name}'")
        print("Press SPACE to capture when ready")
        print("Press 'q' to finish early, 'r' to redo last sample")
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Create overlay for instructions
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Draw hand landmarks with custom styling
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )
            
            # Display instructions and progress
            cv2.putText(frame, f"Samples: {samples_collected}/{num_samples}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture_name}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Gesture Training', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r') and landmarks_data:  # Redo last sample
                landmarks_data.pop()
                samples_collected -= 1
                print("Removed last sample. Redo it!")
            elif key == ord(' '):  # Manual capture with spacebar
                if results.multi_hand_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
                    landmarks_data.append(landmarks)
                    self._save_gesture_image(frame, gesture_name, samples_collected)
                    samples_collected += 1
                    print(f"Sample {samples_collected} captured!")
                else:
                    print("No hand detected! Please show your hand clearly.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if landmarks_data:
            self.gesture_data[gesture_name] = landmarks_data
            return True
        return False
    
    def save_gesture_data(self):
        """Save collected gesture data and generate summary"""
        # Create backup before saving
        self._backup_data_file()
        
        # Save binary data
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.gesture_data, f)
        
        print(f"\nGesture data saved to {self.data_file}")
        
        # Save readable summary with more details
        with open(self.summary_file, 'w') as f:
            f.write("Gesture Training Summary\n")
            f.write("=====================\n\n")
            f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Gestures: {len(self.gesture_data)}\n\n")
            
            for gesture, data in self.gesture_data.items():
                f.write(f"Gesture: {gesture}\n")
                f.write(f"Samples: {len(data)}\n")
                f.write(f"Sample Images: {self.image_dir}/{gesture}_sample_*.jpg\n")
                f.write("-----------------\n")
        
        print(f"Summary saved to {self.summary_file}")
        print(f"Gesture images saved in {self.image_dir}")

def main():
    trainer = GestureTrainer()
    
    print("\n🎯 Welcome to Gesture Trainer!")
    print("This program will help you create your own gesture dataset.")
    
    # Show existing gestures if any
    if trainer.gesture_data:
        print("\n📚 Existing gestures:", list(trainer.gesture_data.keys()))
    
    while True:
        print("\n🔍 Options:")
        print("1. Add new gesture")
        print("2. Delete existing gesture")
        print("3. View gesture summary")
        print("4. Validate gestures")
        print("5. Save and exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            gesture_name = input("Enter gesture name (e.g., 'A', 'B', 'C'): ").upper()
            if gesture_name in trainer.gesture_data:
                overwrite = input("⚠️ Gesture already exists. Overwrite? (y/n): ").lower()
                if overwrite != 'y':
                    continue
            
            num_samples = int(input("Enter number of samples to collect (recommended 20): "))
            
            print("\n🎯 Get ready to show your gesture!")
            print("Position your hand clearly in front of the camera.")
            print("Tips:")
            print("- Ensure good lighting")
            print("- Keep your hand steady")
            print("- Maintain consistent gesture form")
            print("- Press SPACE to capture")
            print("- Press R to redo last sample")
            input("Press Enter when ready...")
            
            if trainer.capture_gesture(gesture_name, num_samples):
                print(f"\n✅ Successfully captured {gesture_name}!")
            else:
                print(f"\n❌ Failed to capture {gesture_name}. Please try again.")
        
        elif choice == '2':
            if not trainer.gesture_data:
                print("\n❌ No gestures to delete!")
                continue
                
            print("\n📚 Existing gestures:", list(trainer.gesture_data.keys()))
            gesture_name = input("Enter gesture name to delete: ").upper()
            if gesture_name in trainer.gesture_data:
                confirm = input(f"⚠️ Are you sure you want to delete '{gesture_name}'? (y/n): ").lower()
                if confirm == 'y':
                    del trainer.gesture_data[gesture_name]
                    print(f"✅ Deleted gesture {gesture_name}")
            else:
                print("❌ Gesture not found!")
        
        elif choice == '3':
            if os.path.exists(trainer.summary_file):
                with open(trainer.summary_file, 'r') as f:
                    print("\n📊 Gesture Summary:")
                    print(f.read())
            else:
                print("\n❌ No summary file found!")
        
        elif choice == '4':
            if not trainer.gesture_data:
                print("\n❌ No gestures to validate!")
                continue
            
            print("\n🔍 Starting gesture validation...")
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = trainer.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
                    
                    # Check similarity with all gestures
                    similarities = {}
                    for gesture, samples in trainer.gesture_data.items():
                        similarity = np.mean([np.mean(np.abs(np.array(landmarks) - np.array(sample))) 
                                           for sample in samples])
                        similarities[gesture] = 1 - similarity  # Convert distance to similarity
                    
                    # Display top 3 matches
                    y_pos = 30
                    for gesture, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]:
                        color = (0, 255, 0) if score > 0.8 else (0, 165, 255)
                        cv2.putText(frame, f"{gesture}: {score:.2f}",
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, color, 2)
                        y_pos += 30
                
                cv2.imshow('Gesture Validation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '5':
            if trainer.gesture_data:
                trainer.save_gesture_data()
            print("\n👋 Thank you for using Gesture Trainer!")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
