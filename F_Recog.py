import face_recognition
import cv2
import pickle
import numpy as np

# load saved encodings
with open("encodings1.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# start webcam
video = cv2.VideoCapture(1)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # detect faces
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):

        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        name = "no data"

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]

        top, right, bottom, left = location

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("Student Recognition", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()