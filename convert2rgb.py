import os
import cv2

# Settings
source_folder = 'faces_resized'
output_folder = 'fixed_photos' # New folder name

# Create the new main folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for folder_name in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, folder_name)
    
    if os.path.isdir(subfolder_path):
        # Create the corresponding subfolder (x, y, z) in 'fixed_photos'
        target_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)

        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(file_path)

            if img is not None:
                # Fix the image
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Save to the NEW location
                save_path = os.path.join(target_subfolder, filename)
                cv2.imwrite(save_path, img)
                print(f"Saved to: {save_path}")