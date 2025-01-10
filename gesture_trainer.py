import cv2
import numpy as np
import os
import mediapipe as mp
import pickle

class GestureTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create directory for storing gesture data
        self.data_dir = 'gesture_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Fixed filenames for gesture data
        self.data_file = os.path.join(self.data_dir, 'gesture_data.pkl')
        self.summary_file = os.path.join(self.data_dir, 'gesture_summary.txt')
        
        # Load existing data if available
        self.gesture_data = self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing gesture data if available"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    return pickle.load(f)
            except:
                print("Could not load existing data, starting fresh")
        return {}
        
    def capture_gesture(self, gesture_name, num_samples=20):
        """Capture and save hand landmarks for a gesture"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        samples_collected = 0
        landmarks_data = []
        
        print(f"\nCollecting samples for gesture '{gesture_name}'")
        print("Press 'SPACE' to capture a sample, 'q' to finish early")
        
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
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display instructions and progress
            cv2.putText(frame,
                       f"Samples: {samples_collected}/{num_samples}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame,
                       f"Gesture: {gesture_name}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Gesture Training', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space key
                if results.multi_hand_landmarks:
                    # Convert landmarks to list of coordinates
                    landmarks = results.multi_hand_landmarks[0]
                    landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
                    landmarks_data.append(landmarks_list)
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
        """Save collected gesture data to file"""
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
                f.write(f"Samples: {len(data)}\n")
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
        print("2. Delete existing gesture")
        print("3. Save and exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
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
            else:
                print(f"\nFailed to capture {gesture_name}. Please try again.")
        
        elif choice == '2':
            if not trainer.gesture_data:
                print("\nNo gestures to delete!")
                continue
                
            print("\nExisting gestures:", list(trainer.gesture_data.keys()))
            gesture_name = input("Enter gesture name to delete: ").upper()
            if gesture_name in trainer.gesture_data:
                del trainer.gesture_data[gesture_name]
                print(f"Deleted gesture {gesture_name}")
            else:
                print("Gesture not found!")
        
        elif choice == '3':
            if trainer.gesture_data:
                trainer.save_gesture_data()
            print("\nThank you for using Gesture Trainer!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
