import cv2
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        raise ValueError("No frames provided for video saving.")

    # Get video dimensions from the first frame
    frame_height, frame_width = output_video_frames[0].shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'XVID' or 'mp4v' if needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for frame in output_video_frames:
        # Ensure frame is in the correct format (uint8, 3 channels)
        if not isinstance(frame, np.ndarray) or frame.shape[:2] != (frame_height, frame_width):
            raise ValueError("All frames must be NumPy arrays of the same shape.")

        out.write(frame)

    out.release()