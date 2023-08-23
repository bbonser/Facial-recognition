import cv2
import dlib
import os

# Get the absolute path to the model file
model_path = os.path.abspath('shape_predictor_68_face_landmarks.dat')

# Load the pre-trained face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmarks predictor from dlib
landmark_predictor = dlib.shape_predictor(model_path)

# Load the OpenCV webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_detector(gray)
    
    for face in faces:
        # Get facial landmarks
        landmarks = landmark_predictor(gray, face)
        
        # Draw rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw landmarks on the face
        for point in landmarks.parts():
            cv2.circle(frame, (point.x, point.y), 2, (0, 0, 255), -1)
    
    # Display the resulting frame
    cv2.imshow('Facial Recognition', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
