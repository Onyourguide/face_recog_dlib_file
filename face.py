import face_recognition
import cv2
import numpy as np

def load_known_faces(known_faces_folder):
    # Load known faces from the given folder
    known_faces = []
    image_files = face_recognition.image_files_in_folder(known_faces_folder)
    for file in image_files:
        image = face_recognition.load_image_file(file)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(face_encoding)
    return known_faces

    #image = face_recognition.load_image_file('image.jpg')
    #face_encoding = face_recognition.face_encodings(image)[0]
    #known_faces.append(face_encoding)
    #return known_faces


def main():
    # Load known faces
    known_faces_folder = "known_faces_a"
    known_faces = load_known_faces(known_faces_folder)

    # Start video capture from default camera (0)
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture each frame
        ret, frame = video_capture.read()

        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the current face matches any known face
            matches = face_recognition.compare_faces(known_faces, face_encoding)

            name = "Unknown"

            # Check if we found a match
            if True in matches:
                first_match_index = matches.index(True)
                name = f"Person {first_match_index + 1}"

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
