import cv2
import numpy as np
import imutils
from imutils.video import VideoStream, FPS

# Gaze tracking function
def gaze_direction(face, eyes):
    # Ensure at least two eyes are detected
    if len(eyes) != 2:
        return "Uncertain"

    # Get the face boundaries
    face_x, face_y, face_w, face_h = face

    # Calculate the midpoint between the eyes
    eye1 = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
    eye2 = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
    eye_center_x = (eye1[0] + eye2[0]) // 2  # Average of the eyes' x-coordinates
    face_center_x = face_x + face_w // 2
    print(eye_center_x)

    # Determine gaze direction based on the relative position of the eye centers
    if eye_center_x < face_center_x - face_w * 0.5:
        return "Looking Straight"
    elif eye_center_x > 60:
        return "Looking Right"
    else:
        return "Looking Straight"

# Main loop for webcam feed
def main():
    vs = VideoStream(src=0).start()
    fps = FPS().start()

    # Load the face and eye classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        frame = vs.read()  # Read the webcam feed
        frame = imutils.resize(frame, width=400)  # Resize for better performance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]  # Region of interest for face
            roi_color = frame[y:y + h, x:x + w]  # Corresponding color region

            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) >= 2:
                # Determine gaze direction
                gaze = gaze_direction((x, y, w, h), eyes[:2])  # Take the first two eyes
                print("Head & Eye Direction:", gaze)

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Draw rectangles around the eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Display the frame with drawn rectangles and gaze direction text
        cv2.imshow("Frame", frame)

        # Break loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        fps.update()  # Update frames per second

    fps.stop()  # Stop FPS tracking
    cv2.destroyAllWindows()  # Close all OpenCV windows
    vs.stop()  # Stop the webcam feed

# Run the main loop
main()  # Start the gaze tracking
