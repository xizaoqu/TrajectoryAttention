import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    target_size = (1024, 576)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, target_size)
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, resized_frame)
        
        frame_count += 1

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_folder)
