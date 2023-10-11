import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import os
import numpy as np

import pandas as pd
rnamelst = pd.read_csv("namelist.csv")
rnamelst = pd.DataFrame(rnamelst)

def load_known_faces(known_faces_folder):
    # Load known faces from the given folder
    known_faces = []
    image_files = face_recognition.face_recognition_cli.image_files_in_folder(known_faces_folder)
    for file in image_files:
        image = face_recognition.load_image_file(file)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(face_encoding)
    return known_faces

def main1():
    # Load known faces
    known_faces_folder = "known_face_folder"
    known_faces = load_known_faces(known_faces_folder)

    # Start video capture from default camera (0)
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture each frame
        ret, frame = video_capture.read()

        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Creat Folder "capture" if not foud
        capture_folder = "capture"
        if not os.path.exists(capture_folder):
            os.makedirs(capture_folder)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the current face matches any known face
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            is_known = any(matches)

            # Check if we found a match
            if True in matches:
                first_match_index = matches.index(True)
                name = rnamelst.iloc[first_match_index,0] #iloc type rnamslst = dataframe array2D firstmatch , colum 0 = name

            if is_known:
                print(True)
                cv2.imwrite(os.path.join(capture_folder, "captured_face.jpg"), frame)
            else:
                print(False)
                cv2.imwrite(os.path.join(capture_folder, "captured_face.jpg"), frame)

            if name == "Unknown":
                color = (0,0,255)
            else :
                color = (0,255,0)
            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (color), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (color), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main1()

