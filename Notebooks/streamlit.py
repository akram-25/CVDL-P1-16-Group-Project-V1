# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import mediapipe as mp
import cv2

st.set_page_config(page_title="Python AR Try-On Demo")
st.title("Python AR Try-On Demo (Snapshot mode)")
st.write("Take a photo or upload an image. Select an outfit to try on.")

uploaded = st.file_uploader("Upload an image (optional)", type=["jpg","png","jpeg"])
cam_image = st.camera_input("Or take a picture")

# Outfit selection
outfit = st.selectbox("Choose outfit", ["red_tshirt", "blue_jacket"])
CLOTHES = {"red_tshirt": "assets/red_tshirt.png", "blue_jacket": "assets/blue_jacket.png"}

def process_image(image_bytes, outfit_path):
    # Convert to OpenCV format
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape
    
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        
        if not res.pose_landmarks:
            return image
        
        # Process landmarks and overlay clothing (similar to above)
        # ... [landmark processing code] ...
        
        return processed_image

# Process and display result
if cam_image or uploaded:
    bytes_data = cam_image.getvalue() if cam_image else uploaded.read()
    result_img = process_image(bytes_data, CLOTHES[outfit])
    st.image(result_img, caption="Try-On Result", use_column_width=True)
