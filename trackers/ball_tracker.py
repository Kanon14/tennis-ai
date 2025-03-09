from ultralytics import YOLO
import cv2
import pickle

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as stub_file:
                ball_detections = pickle.load(stub_file)
            return ball_detections
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
            
        if stub_path is not None:
            with open(stub_path, "wb") as stub_file:
                pickle.dump(ball_detections, stub_file)
            
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            
            ball_dict[1] = result
                
        return ball_dict
 
    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw the bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33, 162, 234), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (33, 162, 234), 2)
            
            output_video_frames.append(frame)
        
        return output_video_frames
                   