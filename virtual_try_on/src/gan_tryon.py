import cv2
import numpy as np
from PIL import Image

class LightweightGANTryOn:
    def __init__(self):
        self.initialized = True
    
    def style_transfer(self, person_frame, garment_img, key_points):
        if not key_points:
            return person_frame
        
        left_shoulder = key_points['left_shoulder']
        right_shoulder = key_points['right_shoulder']
        left_hip = key_points['left_hip']
        right_hip = key_points['right_hip']
        
        width = abs(right_shoulder[0] - left_shoulder[0]) * 2.2
        height = abs((left_shoulder[1] + right_shoulder[1])//2 - (left_hip[1] + right_hip[1])//2) * 1.8
        
        if width <= 0 or height <= 0:
            return person_frame
        
        garment_resized = cv2.resize(garment_img, (int(width), int(height)))
        
        center_x = (left_shoulder[0] + right_shoulder[0]) // 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2
        
        x = center_x - int(width // 2)
        y = shoulder_y - int(height * 0.1)
        
        result = person_frame.copy()
        
        x = max(0, x)
        y = max(0, y)
        
        if garment_resized.shape[2] == 4:
            overlay_rgb = garment_resized[:, :, :3]
            alpha = garment_resized[:, :, 3] / 255.0
            
            x_end = min(result.shape[1], x + garment_resized.shape[1])
            y_end = min(result.shape[0], y + garment_resized.shape[0])
            
            overlay_width = x_end - x
            overlay_height = y_end - y
            
            if overlay_width > 0 and overlay_height > 0:
                overlay_rgb = overlay_rgb[:overlay_height, :overlay_width]
                alpha = alpha[:overlay_height, :overlay_width]
                
                roi = result[y:y_end, x:x_end]
                
                # Adaptive blending based on lighting
                brightness_factor = cv2.mean(roi)[0] / 128.0
                overlay_rgb_adjusted = cv2.convertScaleAbs(overlay_rgb, 
                                                          alpha=brightness_factor, 
                                                          beta=0)
                
                # Blend with alpha
                for c in range(3):
                    roi[:, :, c] = (alpha * overlay_rgb_adjusted[:, :, c] + 
                                   (1 - alpha) * roi[:, :, c])
                
                result[y:y_end, x:x_end] = roi
        
        return result
