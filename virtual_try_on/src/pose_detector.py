import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                landmarks.append([int(landmark.x * w), int(landmark.y * h)])
        
        return landmarks, results
    
    def get_key_points(self, landmarks):
        if len(landmarks) < 33:
            return None
        
        key_points = {
            'left_shoulder': landmarks[11],
            'right_shoulder': landmarks[12],
            'left_elbow': landmarks[13],
            'right_elbow': landmarks[14],
            'left_hip': landmarks[23],
            'right_hip': landmarks[24]
        }
        return key_points
