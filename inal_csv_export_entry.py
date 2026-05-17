import face_recognition
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# load saved encodings
with open("encodings1.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# initialize attendance
attendance = {}
session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Session started at: {session_start}")

# start webcam
video = cv2.VideoCapture(1)

if not video.isOpened():
    print("Camera index 1 not found, trying index 0...")
    video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: No camera found. Exiting.")
    exit()

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

        name = "Unknown"

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                name = known_names[best_match_index]

                # mark attendance only once per person
                if name not in attendance:
                    attendance[name] = datetime.now().strftime("%H:%M:%S")
                    print(f"Marked present: {name} at {attendance[name]}")

        top, right, bottom, left = location

        # green box for known, red box for unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    # show live attendance count on screen
    cv2.putText(frame, f"Present: {len(attendance)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Student Attendance System", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

# export attendance to CSV
date_today = datetime.now().strftime("%Y-%m-%d")

if len(attendance) == 0:
    print("Warning: No faces were recognized. CSV will be empty.")

df = pd.DataFrame(list(attendance.items()), columns=["Name", "Time"])
df["Date"] = date_today
df["Status"] = "Present"
df = df[["Name", "Date", "Time", "Status"]]

filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(filename, index=False)

print(f"\nAttendance Summary:")
print(df.to_string(index=False))
print(f"\nSaved to: {filename}")