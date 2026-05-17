import cv2
import os

video_folder = "videos"
output_root = "frames"

os.makedirs(output_root, exist_ok=True)

for video_file in os.listdir(video_folder):
    
    video_path = os.path.join(video_folder, video_file)
    video_name = os.path.splitext(video_file)[0]

    output_folder = os.path.join(output_root, video_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    interval = max(int(fps / 10), 1)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_path = os.path.join(
                output_folder, f"{video_name}_{saved_count:06d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"{video_file}: saved {saved_count} frames")

print("Done.")