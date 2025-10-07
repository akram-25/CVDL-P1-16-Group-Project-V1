import cv2
import numpy as np

def overlay_transparent_image(background, overlay, x, y):
    """
    Overlay a transparent PNG image onto a background image
    """
    # Ensure coordinates are within bounds
    bg_h, bg_w = background.shape[:2]
    overlay_h, overlay_w = overlay.shape[:2]
    
    if x < 0 or y < 0 or x + overlay_w > bg_w or y + overlay_h > bg_h:
        # Adjust overlay to fit within background
        x = max(0, min(x, bg_w - overlay_w))
        y = max(0, min(y, bg_h - overlay_h))
    
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
        background[y:y+overlay_h, x:x+overlay_w] = overlay
    
    return background

def create_shirt_mask(shirt_image):
    """
    Create a mask for the shirt image for better blending
    """
    if shirt_image.shape[2] == 4:
        return shirt_image[:, :, 3]
    else:
        gray = cv2.cvtColor(shirt_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return mask
