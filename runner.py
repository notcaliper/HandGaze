import cv2
import face_recognition

# Load Known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces and their names here
known_faces = {
    "AKSHAY": 'akshay.jpg',
    "CHETAN": 'chetan.jpg',
    "AJAY": 'ajay.jpg',
    "GAURAV": 'gaurav.jpg'
}

for name, image_path in known_faces.items():
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# Initialize webcam with a lower resolution
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # Set width
video_capture.set(4, 480)  # Set height

frame_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame_counter += 1

    # Process every 4th frame to reduce lag
    if frame_counter % 4 == 0:
        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = 'Unknown'

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face and label with the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV Windows
video_capture.release()
cv2.destroyAllWindows()
