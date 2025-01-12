import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import json
from typing import List, Dict, Tuple

class ImageAnalyzer:
    def __init__(self, data_dir: str = 'analyzed_data'):
        """Initialize the image analyzer with MediaPipe hands model"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create directory for storing analyzed data
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for better hand detection"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply some basic preprocessing
        image_rgb = cv2.GaussianBlur(image_rgb, (5, 5), 0)
        return image_rgb
    
    def extract_hand_landmarks(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Extract hand landmarks from the image"""
        processed_image = self.preprocess_image(image)
        results = self.hands.process(processed_image)
        
        landmarks_data = []
        annotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                self.mp_draw.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                landmarks_data.append({
                    'landmarks': landmarks,
                    'timestamp': datetime.now().isoformat()
                })
        
        return annotated_image, landmarks_data
    
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze an image and extract hand gesture information"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Extract landmarks
        annotated_image, landmarks_data = self.extract_hand_landmarks(image)
        
        # Save the annotated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(self.data_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_image)
        
        # Prepare analysis results
        analysis_result = {
            'image_path': image_path,
            'annotated_path': annotated_path,
            'image_dimensions': {'height': height, 'width': width},
            'hands_detected': len(landmarks_data),
            'landmarks_data': landmarks_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save analysis data
        analysis_path = os.path.join(self.data_dir, f"{base_name}_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        return analysis_result
    
    def batch_analyze(self, image_dir: str) -> List[Dict]:
        """Analyze all images in a directory"""
        results = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                try:
                    result = self.analyze_image(image_path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        return results

def main():
    # Example usage
    analyzer = ImageAnalyzer()
    
    # Create a test directory if it doesn't exist
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # You can use the analyzer like this:
    # results = analyzer.batch_analyze("path/to/your/images")
    print("Image Analyzer initialized. Ready to process images.")
    print(f"Analyzed data will be saved in: {analyzer.data_dir}")

if __name__ == "__main__":
    main()
