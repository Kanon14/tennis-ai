from ultralytics import YOLO

model = YOLO("yolo12x.pt")

model.track("data/input_video.mp4", conf=0.2, save=True)