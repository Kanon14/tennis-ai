from utils import read_video, save_video
from trackers import PlayerTracker

def main():
    # Read the video file
    input_video_path = "data/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # Detecting players
    player_tracker = PlayerTracker(model_path="models/yolo12x.pt")
    player_detections = player_tracker.detect_frames(video_frames)
    
    # Draw output
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    
    # Save the video
    save_video(output_video_frames, "output_videos/output_video.mp4")
    
if __name__ == "__main__":
    main()