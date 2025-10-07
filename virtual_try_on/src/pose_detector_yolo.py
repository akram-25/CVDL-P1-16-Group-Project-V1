import cv2
import numpy as np
from ultralytics import YOLO

class PoseDetectorYOLO:
    def __init__(self):
        # Load YOLOv8 pose model
        self.model = YOLO('yolov8n-pose.pt')
        print("YOLOv8 pose model loaded successfully")
        
    def detect_pose(self, frame):
        # Run YOLOv8 pose detection
        results = self.model(frame, verbose=False)
        
        landmarks = []
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # Extract keypoints (17 COCO keypoints)
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            # Get first person's keypoints (most confident detection)
            person_keypoints = keypoints[0]
            
            # Convert to pixel coordinates and filter by confidence
            h, w = frame.shape[:2]
            for kp in person_keypoints:
                if kp[2] > 0.3:  # confidence threshold
                    landmarks.append([int(kp[0]), int(kp[1])])
                else:
                    landmarks.append([0, 0])  # placeholder for low confidence points
        
        return landmarks, results[0]
    
    def get_key_points(self, landmarks):
        if len(landmarks) < 17:
            return None
        
        # Check if key points are valid (not [0,0])
        left_shoulder = landmarks[5]
        right_shoulder = landmarks[6]
        left_hip = landmarks[11]
        right_hip = landmarks[12]
        
        # Validate key points
        if (left_shoulder == [0, 0] or right_shoulder == [0, 0] or 
            left_hip == [0, 0] or right_hip == [0, 0]):
            return None
        
        key_points = {
            'left_shoulder': left_shoulder,    # Index 5
            'right_shoulder': right_shoulder,  # Index 6  
            'left_elbow': landmarks[7],        # Index 7
            'right_elbow': landmarks[8],       # Index 8
            'left_hip': left_hip,              # Index 11
            'right_hip': right_hip             # Index 12
        }
        return key_points
