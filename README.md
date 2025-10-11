# CV-P1-16-Group-Project

# Real-Time Virtual Try-On System

An interactive computer vision application that enables real-time virtual try-on of shirts using webcam input. The system employs YOLOv8 pose detection and gesture-based controls for an intuitive, hands-free experience.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

### Core Functionality
- Real-time Pose Detection - YOLOv8-based body landmark tracking
- Dynamic Shirt Overlay - Proportional scaling and positioning based on body measurements
- Gesture Navigation - Hands-free shirt selection using MediaPipe hand tracking
- Upload Custom Shirts - Drag-and-drop or browse to add your own designs
- Background Removal - Automatic background removal using rembg
- Visual Delete Confirmation - In-app dialog for safe shirt deletion

### Advanced Features
- GAN Enhancement (Optional) - AI-powered realistic rendering
- Pose Keypoint Visualization - Toggle skeleton overlay for debugging
- Overlay Toggle - Show/hide virtual try-on effect
- Multi-Selection Interface - Dropdown menu, carousel, and keyboard controls
- Screenshot Capture - Save your favorite looks

### User Interface
- Interactive Carousel - Visual preview of previous, current, and next shirts
- Dropdown Menu - Click-to-select shirt browser with scrolling
- Keyboard Shortcuts - Quick access to all features
- Mouse Support - Click and scroll navigation

---

### Software Requirements
- Python 3.11
- pip (Python package manager)
- Webcam drivers properly installed

---

## Installation

### 1. Clone the Repository

### 2. Create Virtual Environment (Recommended)

Use Python 3.11 for MediaPipe support

### 3. Install Dependencies

Install the following dependencies:
opencv-python
numpy
pillow
mediapipe
ultralytics
torch
torchvision
rembg
onnxruntime
tkinter

### 4. Run Application

Run the main.py file