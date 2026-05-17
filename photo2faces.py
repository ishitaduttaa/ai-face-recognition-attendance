import cv2
import os

input_root = "frames"
output_root = "faces_cropped_50%"
os.makedirs(output_root, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
scale = 1.5  # 50% expansion

for subfolder in os.listdir(input_root):
    input_folder = os.path.join(input_root, subfolder)
    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(output_root, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(faces):
            new_w = int(w * scale)
            new_h = int(h * scale)
            new_x = max(0, x - (new_w - w)//2)
            new_y = max(0, y - (new_h - h)//2)
            new_w = min(new_w, frame.shape[1] - new_x)
            new_h = min(new_h, frame.shape[0] - new_y)

            face_img = frame[new_y:new_y+new_h, new_x:new_x+new_w]

            face_name = f"{os.path.splitext(img_file)[0]}_face_{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, face_name), face_img)

print("Face cropping done for all folders!")