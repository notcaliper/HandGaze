**GitHub Repository Description:**

This repository contains the Python script for a simple face recognition application using OpenCV and Face Recognition libraries.

**Repository Structure:**

The repository has the following structure:
```
face-recognition-app/
 runner.py
 requirements.txt
 README.md
```
**Files:**

1. `runner.py`: The main Python script that runs the face recognition application.
2. `requirements.txt`: A text file listing the dependencies required to run the script.
3. `README.md`: A markdown file providing an overview of the repository and its contents.

**Setup Instructions:**

1. Clone the repository using Git: `git clone https://github.com/notcaliper/opencv-hand-gesture-recognition.git`
2. Install the dependencies listed in `requirements.txt` using pip: `pip install -r requirements.txt`
3. Run the script using Python: `python runner.py`

**Usage Instructions:**

1. Open a terminal or command prompt and navigate to the repository directory.
2. Run the script using Python: `python runner.py`
3. A window will appear displaying the webcam feed with face detection and labeling (if found).
4. Press 'q' to exit the loop and close the OpenCV windows.

**Commit Messages and API Documentation:**

The repository follows standard Git commit message conventions and includes API documentation for the script using Markdown comments.

**Known Faces:**

To add new faces or update existing ones, modify the `known_faces.json` file. Each entry should be in the format:
```json
{
    "name": "John Doe",
    "image_path": "path/to/image.jpg"
}
```
Make sure to update the image path if necessary.

