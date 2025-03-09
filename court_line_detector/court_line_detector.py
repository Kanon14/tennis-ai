import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        """
        Initializes the CourtLineDetector model.

        Args:
            model_path (str): Path to the trained model file.
        """
        # Detect available device (CUDA if available, else fallback to CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ResNet50 without pre-trained weights
        self.model = models.resnet50(weights=None)

        # Modify the final fully connected layer for keypoint detection (14 keypoints × 2 coordinates)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)

        # Load the trained model weights with correct device mapping
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Move the model to the specified device
        self.model.to(self.device)

        # Set model to evaluation mode (important for inference)
        self.model.eval()

        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image
            transforms.Resize((224, 224)),  # Resize to match model input size
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
        ])

    def predict(self, image):
        """
        Predicts keypoints for a given image.

        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV).

        Returns:
            numpy.ndarray: Predicted keypoints in the original image scale.
        """
        # Convert BGR (OpenCV format) to RGB (expected format for PyTorch)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations and add batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Perform inference without tracking gradients (reduces memory usage)
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Extract and move keypoints to CPU
        keypoints = outputs.squeeze().cpu().numpy()

        # Rescale keypoints back to the original image dimensions
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0  # Scale x-coordinates
        keypoints[1::2] *= original_h / 224.0  # Scale y-coordinates

        return keypoints

    def draw_keypoints(self, image, keypoints):
        """
        Draws keypoints on an image.

        Args:
            image (numpy.ndarray): Original image.
            keypoints (numpy.ndarray): Array of keypoint coordinates.

        Returns:
            numpy.ndarray: Image with keypoints drawn.
        """
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            # Add a small identifier text above the keypoint
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw a red circle on each keypoint
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draws keypoints on a sequence of video frames.

        Args:
            video_frames (list of numpy.ndarray): List of video frames.
            keypoints (numpy.ndarray): Array of keypoint coordinates for each frame.

        Returns:
            list of numpy.ndarray: Video frames with keypoints drawn.
        """
        output_video_frames = []

        for frame in video_frames:
            frame_with_keypoints = self.draw_keypoints(frame, keypoints)  # Draw keypoints
            output_video_frames.append(frame_with_keypoints)

        return output_video_frames
