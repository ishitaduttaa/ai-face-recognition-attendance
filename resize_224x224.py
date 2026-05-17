import cv2
import os

input_root = "faces_cropped_50%"       # your cropped faces folder
output_root = "faces_resized"      # folder to save resized images
target_size = (224, 224)           # width, height
os.makedirs(output_root, exist_ok=True)

# Loop through subfolders (person1, person2, etc.)
for subfolder in os.listdir(input_root):
    input_folder = os.path.join(input_root, subfolder)
    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(output_root, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize image to 224x224
        resized_img = cv2.resize(img, target_size)

        # Save resized image
        cv2.imwrite(os.path.join(output_folder, img_file), resized_img)

print("All images resized to 224x224!")