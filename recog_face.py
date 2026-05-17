import face_recognition
import os
import pickle
import cv2

dataset_path = "fixed_photos"

known_encodings = []
known_names = []

# loop through each student
for student_name in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_name)

    if not os.path.isdir(student_folder):
        continue
    for image_name in os.listdir(student_folder):

        image_path = os.path.join(student_folder, image_name)

        # image=cv2.imread(image_path)
        # if image is None:
        #     print("Could not read:", image_path)
        #     continue
        # rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        rgb=face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(rgb)

        if len(encodings) == 0:
            print("No face found in", image_name)
            continue

        encoding = encodings[0]

        known_encodings.append(encoding)
        known_names.append(student_name)


data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("encodings1.pkl", "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved successfully!")