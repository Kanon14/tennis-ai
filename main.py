from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    # Read the video file
    input_video_path = "data/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # Detecting players and ball (YOLO)
    player_tracker = PlayerTracker(model_path="models/yolo12x.pt")
    ball_tracker = BallTracker(model_path="models/tennis-ball-yolov12l.pt")
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Detecting court lines (CNNs)
    court_line_detector = CourtLineDetector(model_path="models/keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    # Draw players and ball
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    # Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Save the video
    save_video(output_video_frames, "output_videos/output_video.mp4")
    
if __name__ == "__main__":
    main()