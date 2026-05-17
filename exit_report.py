import face_recognition
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# load saved encodings
with open("encodings1.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# load the existing attendance CSV saved by Part 1
import glob
import os

csv_files = glob.glob("attendance_*.csv")

if len(csv_files) == 0:
    print("Error: No attendance CSV found from Part 1. Run the entry camera first.")
    exit()

# pick the most recent CSV
latest_csv = max(csv_files, key=os.path.getctime)
print(f"Loaded entry attendance from: {latest_csv}")

entry_df = pd.read_csv(latest_csv)
print(entry_df.to_string(index=False))

# initialize exit log
exit_log = {}

# start exit camera
video = cv2.VideoCapture(1)

if not video.isOpened():
    print("Error: Camera index 0 not found.")
    exit()

print("\nExit camera started. Press q to stop.")

while True:
    ret, frame = video.read()

    if not ret:
        break

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

                # record exit time only once
                if name not in exit_log:
                    exit_log[name] = datetime.now().strftime("%H:%M:%S")
                    print(f"Exit recorded: {name} at {exit_log[name]}")

        # draw plain box only, no name or status text on screen
        top, right, bottom, left = location
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    cv2.imshow("Exit Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

# compare entry time and exit time, decide present or absent
date_today = datetime.now().strftime("%Y-%m-%d")
results = []

# get all names from entry CSV
all_entry_names = entry_df["Name"].tolist()

for name in all_entry_names:
    entry_row = entry_df[entry_df["Name"] == name]
    entry_time_str = entry_row["Time"].values[0]

    if name in exit_log:
        exit_time_str = exit_log[name]

        # calculate time difference
        entry_time = datetime.strptime(entry_time_str, "%H:%M:%S")
        exit_time  = datetime.strptime(exit_time_str,  "%H:%M:%S")
        diff = exit_time - entry_time

        minutes_present = diff.total_seconds() / 60

        if minutes_present >= 40:
            status = "Present"
        else:
            status = "Absent"

        results.append({
            "Name":       name,
            "Date":       date_today,
            "Entry Time": entry_time_str,
            "Exit Time":  exit_time_str,
            "Duration (min)": round(minutes_present, 1),
            "Status":     status
        })

    else:
        # person entered but never passed exit camera
        results.append({
            "Name":       name,
            "Date":       date_today,
            "Entry Time": entry_time_str,
            "Exit Time":  "Not recorded",
            "Duration (min)": 0,
            "Status":     "Absent"
        })

# also handle names that appear at exit but were never in entry CSV
for name in exit_log:
    if name not in all_entry_names:
        results.append({
            "Name":       name,
            "Date":       date_today,
            "Entry Time": "Not recorded",
            "Exit Time":  exit_log[name],
            "Duration (min)": 0,
            "Status":     "Absent"
        })

# build final dataframe
final_df = pd.DataFrame(results, columns=[
    "Name", "Date", "Entry Time", "Exit Time", "Duration (min)", "Status"
])

# save final CSV
final_filename = f"final_attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
final_df.to_csv(final_filename, index=False)

print(f"\nFinal Attendance Report:")
print(final_df.to_string(index=False))
print(f"\nSaved to: {final_filename}")