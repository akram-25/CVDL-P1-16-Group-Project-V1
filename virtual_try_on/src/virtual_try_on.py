import cv2
import numpy as np
import os
from PIL import Image

class VirtualTryOn:
    def __init__(self, shirts_folder="../assets/shirts/"):
        self.shirts_folder = shirts_folder
        self.shirt_images = self.load_shirt_images()
        self.current_shirt_index = 0 if self.shirt_images else -1
        
    def load_shirt_images(self):
        shirts = []
        shirts_path = os.path.join(os.path.dirname(__file__), self.shirts_folder)
        
        if os.path.exists(shirts_path):
            for filename in os.listdir(shirts_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(shirts_path, filename)
                    try:
                        # Load with PIL first to handle transparency properly
                        pil_img = Image.open(img_path)
                        if pil_img.mode != 'RGBA':
                            pil_img = pil_img.convert('RGBA')
                        
                        # Convert to OpenCV format
                        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
                        shirts.append((filename, cv_img))
                        print(f"Loaded shirt: {filename}")
                    except Exception as e:
                        print(f"Failed to load {filename}: {e}")
        else:
            print(f"Shirts folder not found: {shirts_path}")
            print("Create the folder and add PNG shirt images")
            
        return shirts
    
    def calculate_shirt_dimensions(self, key_points):
        if not key_points:
            return None
        
        left_shoulder = key_points['left_shoulder']
        right_shoulder = key_points['right_shoulder']
        left_hip = key_points['left_hip']
        right_hip = key_points['right_hip']
        
        # Calculate shoulder width
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Calculate torso height (shoulder to hip)
        torso_height = abs((left_shoulder[1] + right_shoulder[1])//2 - 
                          (left_hip[1] + right_hip[1])//2)
        
        # Scale shirt based on body proportions
        shirt_width = int(shoulder_width * 2.2)  # Make shirt wider than shoulders
        shirt_height = int(max(torso_height * 1.8, shirt_width * 1.2))  # Maintain reasonable aspect ratio
        
        # Calculate position (centered on torso)
        center_x = (left_shoulder[0] + right_shoulder[0]) // 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
        
        return {
            'width': shirt_width,
            'height': shirt_height,
            'x': center_x - shirt_width // 2,
            'y': shoulder_center_y - int(shirt_height * 0.1)  # Slightly above shoulder line
        }
    
    def overlay_shirt(self, frame, key_points):
        if not self.shirt_images or self.current_shirt_index < 0 or not key_points:
            return frame
        
        dimensions = self.calculate_shirt_dimensions(key_points)
        if not dimensions:
            return frame
        
        current_shirt = self.shirt_images[self.current_shirt_index][1]
        
        # Resize shirt to fit person
        resized_shirt = cv2.resize(current_shirt, 
                                 (dimensions['width'], dimensions['height']))
        
        # Overlay the shirt on the frame
        result_frame = self.overlay_transparent_image(frame, resized_shirt, 
                                                    dimensions['x'], dimensions['y'])
        
        return result_frame
    
    def overlay_transparent_image(self, background, overlay, x, y):
        # Ensure coordinates are within bounds
        bg_h, bg_w = background.shape[:2]
        overlay_h, overlay_w = overlay.shape[:2]
        
        # Adjust if overlay goes outside frame
        if x < 0:
            overlay = overlay[:, -x:]
            overlay_w += x
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            overlay_h += y
            y = 0
        if x + overlay_w > bg_w:
            overlay = overlay[:, :bg_w-x]
            overlay_w = bg_w - x
        if y + overlay_h > bg_h:
            overlay = overlay[:bg_h-y, :]
            overlay_h = bg_h - y
        
        if overlay_w <= 0 or overlay_h <= 0:
            return background
        
        # Extract alpha channel if available
        if overlay.shape[2] == 4:
            # Split the overlay into color and alpha channels
            overlay_rgb = overlay[:, :, :3]
            overlay_alpha = overlay[:, :, 3] / 255.0
            
            # Get the region of interest from background
            roi = background[y:y+overlay_h, x:x+overlay_w]
            
            # Blend the images using alpha channel
            for c in range(3):
                roi[:, :, c] = (overlay_alpha * overlay_rgb[:, :, c] + 
                               (1 - overlay_alpha) * roi[:, :, c])
            
            background[y:y+overlay_h, x:x+overlay_w] = roi
        else:
            # Simple overlay without transparency
            background[y:y+overlay_h, x:x+overlay_w] = overlay[:, :, :3]
        
        return background
    
    def next_shirt(self):
        if self.shirt_images:
            self.current_shirt_index = (self.current_shirt_index + 1) % len(self.shirt_images)
    
    def previous_shirt(self):
        if self.shirt_images:
            self.current_shirt_index = (self.current_shirt_index - 1) % len(self.shirt_images)
    
    def get_current_shirt_name(self):
        if self.shirt_images and self.current_shirt_index >= 0:
            return self.shirt_images[self.current_shirt_index][0]
        return "No shirts available"
