import cv2
import os
import numpy as np
from image_analyzer import ImageAnalyzer

def create_sample_image():
    """Create a simple test image with a colored rectangle"""
    # Create a black image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a colored rectangle (simulating a hand)
    cv2.rectangle(img, (200, 150), (400, 350), (0, 255, 0), -1)
    
    # Create test_images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Save the test image
    image_path = os.path.join('test_images', 'sample_hand.jpg')
    cv2.imwrite(image_path, img)
    print(f"Created sample image at: {image_path}")
    return image_path

def main():
    # Initialize the analyzer
    analyzer = ImageAnalyzer()
    
    # 1. First, let's create a sample image
    print("Step 1: Creating a sample image...")
    image_path = create_sample_image()
    
    # 2. Analyze the single image
    print("\nStep 2: Analyzing the image...")
    try:
        result = analyzer.analyze_image(image_path)
        print("\nAnalysis Results:")
        print(f"- Hands detected: {result['hands_detected']}")
        print(f"- Annotated image saved to: {result['annotated_path']}")
        print(f"- Analysis data saved to: {os.path.splitext(result['annotated_path'])[0]}_analysis.json")
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
    
    # 3. Show the annotated image
    print("\nStep 3: Displaying the analyzed image...")
    annotated_image = cv2.imread(result['annotated_path'])
    if annotated_image is not None:
        cv2.imshow('Analyzed Image', annotated_image)
        print("Press any key to close the image window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
