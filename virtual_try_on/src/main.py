import cv2
import sys
import os
import time
import mediapipe as mp
import numpy as np
from PIL import Image
from pose_detector_yolo import PoseDetectorYOLO
from virtual_try_on import VirtualTryOn
from upload_manager import ShirtUploadManager
from gan_tryon import LightweightGANTryOn

# Custom dropdown menu class
class ShirtDropdown:
    def __init__(self, position=(10, 210), width=300, item_height=40):
        self.position = position
        self.width = width
        self.item_height = item_height
        self.max_visible_items = 8
        self.is_open = False
        self.hover_index = -1
        self.scroll_offset = 0

    # Draw the dropdown menu
    def draw(self, frame, shirt_names, current_index):
        if not shirt_names:
            return frame
        
        x, y = self.position
        
        # Draw main button (closed state)
        if not self.is_open:
            self._draw_closed_button(frame, shirt_names, current_index, x, y)
        else:
            self._draw_open_menu(frame, shirt_names, current_index, x, y)
        
        return frame
    
    def _draw_closed_button(self, frame, shirt_names, current_index, x, y):
        # Background
        cv2.rectangle(frame, (x, y), (x + self.width, y + self.item_height),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + self.width, y + self.item_height),
                     (0, 255, 0), 2)
        
        # Current shirt name
        text = shirt_names[current_index]
        if len(text) > 30:
            text = text[:27] + "..."
        
        cv2.putText(frame, text, (x + 10, y + 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        arrow_x = x + self.width - 30
        arrow_y = y + self.item_height // 2
        
        pts = np.array([
            [arrow_x, arrow_y - 5],
            [arrow_x - 8, arrow_y - 12],
            [arrow_x + 8, arrow_y - 12]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (255, 255, 255))
        
        count_text = f"{current_index + 1}/{len(shirt_names)}"
        cv2.putText(frame, count_text, (x + self.width - 80, y + 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_open_menu(self, frame, shirt_names, current_index, x, y):
        total_items = len(shirt_names)
        visible_items = min(self.max_visible_items, total_items)
        
        menu_height = visible_items * self.item_height
        
        # Solid background
        cv2.rectangle(frame, (x, y), 
                    (x + self.width, y + self.item_height + menu_height),
                    (40, 40, 40), -1)
        
        # Green border
        cv2.rectangle(frame, (x, y), 
                    (x + self.width, y + self.item_height + menu_height),
                    (0, 255, 0), 3)
        
        # Current selection
        cv2.rectangle(frame, (x, y), (x + self.width, y + self.item_height),
                    (70, 70, 70), -1)
        
        text = shirt_names[current_index]
        if len(text) > 30:
            text = text[:27] + "..."
        cv2.putText(frame, text, (x + 10, y + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        arrow_x = x + self.width - 30
        arrow_y = y + self.item_height // 2
        pts = np.array([
            [arrow_x, arrow_y + 5],
            [arrow_x - 8, arrow_y + 12],
            [arrow_x + 8, arrow_y + 12]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (255, 255, 255))
        
        end_index = min(self.scroll_offset + visible_items, total_items)
        
        for i in range(self.scroll_offset, end_index):
            item_y = y + self.item_height + (i - self.scroll_offset) * self.item_height
            
            item_color = (60, 60, 60)
            
            # Highlight current selection in blue
            if i == current_index:
                item_color = (200, 100, 50)  # Orange for current
            
            if i == self.hover_index:
                item_color = (50, 200, 100)  # Green for hover
            
            cv2.rectangle(frame, (x + 2, item_y + 2),
                        (x + self.width - 2, item_y + self.item_height - 2),
                        item_color, -1)
            
            # Border around each item
            cv2.rectangle(frame, (x + 2, item_y + 2),
                        (x + self.width - 2, item_y + self.item_height - 2),
                        (100, 100, 100), 1)
            
            # Item text
            item_text = f"{i+1}. {shirt_names[i]}"
            if len(item_text) > 32:
                item_text = item_text[:29] + "..."
            
            text_color = (255, 255, 255)
            cv2.putText(frame, item_text, (x + 10, item_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        if total_items > visible_items:
            self._draw_scrollbar(frame, x, y, total_items, visible_items)
        
        return frame

    
    def _draw_scrollbar(self, frame, x, y, total_items, visible_items):
        scrollbar_x = x + self.width - 15
        scrollbar_y_start = y + self.item_height + 5
        scrollbar_height = visible_items * self.item_height - 10
        
        cv2.rectangle(frame, 
                     (scrollbar_x, scrollbar_y_start),
                     (scrollbar_x + 8, scrollbar_y_start + scrollbar_height),
                     (100, 100, 100), -1)
        
        thumb_height = max(20, int(scrollbar_height * visible_items / total_items))
        thumb_y = scrollbar_y_start + int(scrollbar_height * self.scroll_offset / total_items)
        
        cv2.rectangle(frame,
                     (scrollbar_x, thumb_y),
                     (scrollbar_x + 8, thumb_y + thumb_height),
                     (0, 255, 0), -1)
    
    def handle_click(self, mouse_x, mouse_y, shirt_count, current_index):
        x, y = self.position
        if (x <= mouse_x <= x + self.width and 
            y <= mouse_y <= y + self.item_height):
            self.is_open = not self.is_open
            return None
        
        if self.is_open:
            visible_items = min(self.max_visible_items, shirt_count)
            end_index = min(self.scroll_offset + visible_items, shirt_count)
            
            for i in range(self.scroll_offset, end_index):
                item_y = y + self.item_height + (i - self.scroll_offset) * self.item_height
                
                if (x <= mouse_x <= x + self.width and
                    item_y <= mouse_y <= item_y + self.item_height):
                    self.is_open = False
                    self.hover_index = -1
                    return i
        
        if self.is_open:
            self.is_open = False
            self.hover_index = -1
        
        return None

    
    def handle_hover(self, mouse_x, mouse_y, shirt_count):
        if not self.is_open:
            self.hover_index = -1
            return
        
        x, y = self.position
        visible_items = min(self.max_visible_items, shirt_count)
        end_index = min(self.scroll_offset + visible_items, shirt_count)
        
        self.hover_index = -1
        for i in range(self.scroll_offset, end_index):
            item_y = y + self.item_height + (i - self.scroll_offset) * self.item_height
            
            if (x <= mouse_x <= x + self.width and
                item_y <= mouse_y <= item_y + self.item_height):
                self.hover_index = i
                break
    
    def handle_scroll(self, delta, shirt_count):
        if not self.is_open:
            return
        
        self.scroll_offset += delta
        max_scroll = max(0, shirt_count - self.max_visible_items)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

class ConfirmationDialog:
    def __init__(self):
        self.is_showing = False
        self.message = ""
        self.title = ""
        self.confirmed = None  # Store result here
        self.yes_button_bounds = None
        self.no_button_bounds = None
        
    def show(self, frame, title, message):
        self.is_showing = True
        self.title = title
        self.message = message
        
        frame_h, frame_w = frame.shape[:2]
        
        # Dimensions
        dialog_w = 500
        dialog_h = 250
        dialog_x = (frame_w - dialog_w) // 2
        dialog_y = (frame_h - dialog_h) // 2
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Dialog background
        cv2.rectangle(frame, (dialog_x, dialog_y), 
                     (dialog_x + dialog_w, dialog_y + dialog_h),
                     (40, 40, 40), -1)
        
        # Dialog border
        cv2.rectangle(frame, (dialog_x, dialog_y), 
                     (dialog_x + dialog_w, dialog_y + dialog_h),
                     (0, 255, 0), 3)
        
        # Title bar
        cv2.rectangle(frame, (dialog_x, dialog_y), 
                     (dialog_x + dialog_w, dialog_y + 50),
                     (60, 60, 60), -1)
        
        # Title text
        cv2.putText(frame, title, (dialog_x + 20, dialog_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Message text
        lines = self._wrap_text(message, 45)
        y_offset = dialog_y + 90
        for line in lines:
            cv2.putText(frame, line, (dialog_x + 30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            y_offset += 30
        
        # Buttons
        button_y = dialog_y + dialog_h - 70
        button_h = 50
        
        # Yes button
        yes_button_x = dialog_x + 80
        yes_button_w = 150
        cv2.rectangle(frame, (yes_button_x, button_y),
                     (yes_button_x + yes_button_w, button_y + button_h),
                     (0, 180, 0), -1)
        cv2.rectangle(frame, (yes_button_x, button_y),
                     (yes_button_x + yes_button_w, button_y + button_h),
                     (0, 255, 0), 2)
        cv2.putText(frame, "YES (Y)", (yes_button_x + 30, button_y + 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # No button
        no_button_x = dialog_x + 270
        no_button_w = 150
        cv2.rectangle(frame, (no_button_x, button_y),
                     (no_button_x + no_button_w, button_y + button_h),
                     (0, 0, 180), -1)
        cv2.rectangle(frame, (no_button_x, button_y),
                     (no_button_x + no_button_w, button_y + button_h),
                     (0, 0, 255), 2)
        cv2.putText(frame, "NO (N)", (no_button_x + 35, button_y + 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Store button positions
        self.yes_button_bounds = (yes_button_x, button_y, yes_button_w, button_h)
        self.no_button_bounds = (no_button_x, button_y, no_button_w, button_h)
        
        return frame
    
    def _wrap_text(self, text, max_chars):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def handle_click(self, mouse_x, mouse_y):
        if not self.is_showing:
            return None
        
        # Check Yes button
        if self.yes_button_bounds:
            x, y, w, h = self.yes_button_bounds
            if x <= mouse_x <= x + w and y <= mouse_y <= y + h:
                self.is_showing = False
                self.confirmed = True
                print("YES clicked")
                return True
        
        # Check No button
        if self.no_button_bounds:
            x, y, w, h = self.no_button_bounds
            if x <= mouse_x <= x + w and y <= mouse_y <= y + h:
                self.is_showing = False
                self.confirmed = False
                print("NO clicked")
                return False
        
        return None

    
    def handle_keypress(self, key):
        if not self.is_showing:
            return None
        
        if key == ord('y') or key == ord('Y'):
            self.is_showing = False
            return True
        elif key == ord('n') or key == ord('N') or key == 27: # Esc key
            self.is_showing = False
            return False
        
        return None
    
    def close(self):
        self.is_showing = False
        self.confirmed = None

class VirtualTryOnApp:
    def __init__(self):
        self.cap = None
        self.gan_tryon = None
        self.hands = None
        self.running = False
        
        try:
            self.pose_detector = PoseDetectorYOLO()
            self.virtual_try_on = VirtualTryOn()
            self.upload_manager = ShirtUploadManager()
            self.current_frame = None
            self.display_frame = None
            
            # MediaPipe
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            
            # GAN Try-On
            self.use_gan = True
            try:
                self.gan_tryon = LightweightGANTryOn()
            except Exception as e:
                self.use_gan = False
                self.gan_tryon = None
            
            # Camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open video capture")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.running = True
            self.last_hand_action = None
            
            # Keypoint visibility toggle
            self.show_keypoints = False
            
            # Shirt overlay toggle
            self.show_shirt_overlay = True
            self.dropdown = ShirtDropdown(position=(10, 400), width=350, item_height=35)

            # Initialize confirmation dialog (NEW)
            self.confirmation_dialog = ConfirmationDialog()
            
            # Define window name
            self.window_name = 'Virtual Try On Application'
            
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
            
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
        except Exception as e:
            print(f" Initialisation failed: {e}")
            if self.cap is not None:
                self.cap.release()
            if self.hands is not None:
                self.hands.close()
            raise


    
    def draw_keypoints(self, frame, landmarks, key_points):
        if not self.show_keypoints:
            return frame
        
        if landmarks is None:
            return frame
        
        if key_points is None or not key_points:
            return frame
        
        try:
            # Skeleton connections
            connections = [
                # Torso
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                
                # Left arm
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                
                # Right arm
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                
                # Left leg
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                
                # Right leg
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle'),
                
                # Head
                ('nose', 'left_eye'),
                ('nose', 'right_eye'),
                ('left_eye', 'left_ear'),
                ('right_eye', 'right_ear'),
            ]
            
            # Draw connections
            for connection in connections:
                point1_name, point2_name = connection
                
                # Check if both points exist
                if point1_name in key_points and point2_name in key_points:
                    point1 = key_points[point1_name]
                    point2 = key_points[point2_name]
                    
                    # Point validation
                    if point1 is not None and point2 is not None:
                        if len(point1) >= 2 and len(point2) >= 2:
                            cv2.line(frame, 
                                    (int(point1[0]), int(point1[1])),
                                    (int(point2[0]), int(point2[1])),
                                    (0, 255, 0), 2)
            
            # Draw keypoints as circles
            for name, point in key_points.items():
                # Validate point
                if point is None or len(point) < 2:
                    continue
                
                try:
                    x, y = int(point[0]), int(point[1])
                    
                    # Skip if coordinates are invalid
                    if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
                        continue
                    
                    # Color code different body parts
                    if 'shoulder' in name or 'hip' in name:
                        color = (255, 0, 0)  # Blue for torso
                        radius = 6
                    elif 'elbow' in name or 'wrist' in name:
                        color = (0, 255, 255)  # Yellow for arms
                        radius = 5
                    elif 'knee' in name or 'ankle' in name:
                        color = (255, 0, 255)  # Magenta for legs
                        radius = 5
                    else:  # Head points
                        color = (0, 0, 255)  # Red for head
                        radius = 4
                    
                    # Draw circle for keypoint
                    cv2.circle(frame, (x, y), radius, color, -1)
                    cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 1)
                    
                    # Draw keypoint label
                    label = name.replace('_', ' ').title()

                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                    cv2.rectangle(frame, 
                                (x + 10, y - 15), 
                                (x + 10 + text_size[0], y - 5),
                                (0, 0, 0), -1)
                    cv2.putText(frame, label, (x + 10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                except Exception as e:
                    continue
        
        except Exception as e:
            print(f"Error drawing keypoints: {e}")
        
        return frame

    
    def detect_hand_raised(self, frame, key_points):
        if self.hands is None:
            return False, False
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            left_up = False
            right_up = False
            frame_height, frame_width = frame.shape[:2]
            
            # Get reference point
            if key_points and 'left_shoulder' in key_points and 'right_shoulder' in key_points:
                # Use shoulder midpoint
                shoulder_mid_y = (key_points['left_shoulder'][1] + key_points['right_shoulder'][1]) / 2
                reference_y = shoulder_mid_y - 50  # 50 pixels above shoulders
            elif key_points and 'nose' in key_points:
                # Use nose as fallback
                reference_y = key_points['nose'][1] + 30  # Slightly below nose
            else:
                # Use upper third of frame as last resort
                reference_y = frame_height * 0.35
            
            # Check detected hands using MediaPipe
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Get hand label (Left or Right from camera perspective)
                    hand_label = handedness.classification[0].label
                    
                    # Get hand position (use middle finger tip and wrist for better accuracy)
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Convert to pixel coordinates
                    wrist_y = wrist.y * frame_height
                    middle_tip_y = middle_tip.y * frame_height
                    index_tip_y = index_tip.y * frame_height
                    
                    # Average of hand points for more stable detection
                    hand_center_y = (wrist_y + middle_tip_y + index_tip_y) / 3
                    
                    # Check if hand is raised (hand center above reference)
                    # Also check that fingers are above wrist (hand is upright)
                    fingers_above_wrist = (middle_tip_y < wrist_y) and (index_tip_y < wrist_y)
                    hand_is_raised = (hand_center_y < reference_y) and fingers_above_wrist
                    
                    # Mirrored logic: Right hand is on left side of image
                    if hand_label == "Right":  # Actual LEFT hand
                        left_up = hand_is_raised
                    elif hand_label == "Left":  # Actual RIGHT hand
                        right_up = hand_is_raised
                    
                    if self.show_keypoints:
                        # Draw hand landmarks
                        wrist_x = int(wrist.x * frame_width)
                        wrist_y_int = int(wrist_y)
                        
                        # Draw circle at wrist
                        color = (0, 255, 0) if hand_is_raised else (0, 0, 255)
                        cv2.circle(frame, (wrist_x, wrist_y_int), 8, color, -1)
                        
                        # Draw text
                        label_text = f"{hand_label}: {'UP' if hand_is_raised else 'DOWN'}"
                        cv2.putText(frame, label_text, (wrist_x + 15, wrist_y_int),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return left_up, right_up
            
        except Exception as e:
            print(f"Hand detection error: {e}")
            return False, False


    def handle_hand_gestures(self, left_up, right_up):
        if left_up and self.last_hand_action != "left_up":
            print("Right hand raised - Next shirt")
            self.virtual_try_on.next_shirt()
            print(f"Switched to: {self.virtual_try_on.get_current_shirt_name()}")
            self.last_hand_action = "left_up"
            
        elif right_up and self.last_hand_action != "right_up":
            print("Left hand raised - Previous shirt")
            self.virtual_try_on.previous_shirt()
            print(f"Switched to: {self.virtual_try_on.get_current_shirt_name()}")
            self.last_hand_action = "right_up"

        # Reset action   
        elif not left_up and not right_up:
            self.last_hand_action = None
    
    def handle_keypress(self, key):
        if key == ord('q') or key == 27:
            print("\n Shutting down...")
            self.running = False
            cv2.destroyAllWindows()
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            return
        elif key == ord('n'):
            self.virtual_try_on.next_shirt()
            print(f"Switched to: {self.virtual_try_on.get_current_shirt_name()}")

        elif key == ord('p'):
            self.virtual_try_on.previous_shirt()
            print(f"Switched to: {self.virtual_try_on.get_current_shirt_name()}")

        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f'screenshot_{timestamp}.jpg'
            if self.display_frame is not None:
                cv2.imwrite(filename, self.display_frame)
                print(f"📸 Screenshot saved as {filename}")
            else:
                print("❌ No frame available to save")

        elif key == ord('r'):
            self.virtual_try_on.current_shirt_index = 0
            print("Reset to first shirt")

        elif key == ord('u'):
            self.upload_new_shirt()

        elif key == ord('d'):
            self.delete_current_shirt()

        elif key == ord('g'):  # Toggle GAN mode
            if self.gan_tryon:
                self.use_gan = not self.use_gan
                mode = "GAN Enhanced" if self.use_gan else "Standard Overlay"
                print(f"Switched to {mode} mode")
            else:
                print("GAN mode not available")
        elif key == ord('k'):  # Toggle keypoints visibility
            self.show_keypoints = not self.show_keypoints
            status = "ON" if self.show_keypoints else "OFF"
            print(f"Keypoints display: {status}")

        elif key == ord('v'):  # Toggle shirt overlay visibility
            self.show_shirt_overlay = not self.show_shirt_overlay
            status = "ON" if self.show_shirt_overlay else "OFF"
            print(f"Shirt overlay: {status}")

    
    def upload_new_shirt(self):
        # Pause video feed
        was_running = self.running
        self.running = False
        
        # Upload the shirt with background removal
        result = self.upload_manager.upload_shirt()
        
        if result and result[0] is not None:
            uploaded_path, filename = result
            print(f"Shirt uploaded and saved with background removed!")
            print(f"Location: assets/shirts/")
            
            # Manually add this new image to the shirt list without reloading all
            try:
                pil_img = Image.open(uploaded_path)
                if pil_img.mode != 'RGBA':
                    pil_img = pil_img.convert('RGBA')
                
                # Convert to OpenCV format
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
                
                # Append to list of shirts
                self.virtual_try_on.shirt_images.append((filename, cv_img))
                print(f"Added new shirt to collection: {filename}")
                print(f"Total shirts now: {len(self.virtual_try_on.shirt_images)}")
                
            except Exception as e:
                print(f"Error adding new shirt to list: {e}")
                # Reload all shirts if append fails
                self.virtual_try_on.shirt_images = self.virtual_try_on.load_shirt_images()
            
            # Switch to the newly uploaded shirt
            if self.virtual_try_on.shirt_images:
                self.virtual_try_on.current_shirt_index = len(self.virtual_try_on.shirt_images) - 1
                print(f"Switched to: {self.virtual_try_on.get_current_shirt_name()}")
        else:
            print("Upload cancelled or failed")
        
        # Resume video feed
        self.running = was_running
    
    def delete_current_shirt(self):
        if not self.virtual_try_on.shirt_images:
            return
        
        # Get current shirt info
        current_shirt_name = self.virtual_try_on.get_current_shirt_name()
        current_index = self.virtual_try_on.current_shirt_index
        
        print(f"\nRequesting deletion confirmation for: {current_shirt_name}")
        
        # Show dialog and wait for response
        confirmed = self.show_delete_confirmation_dialog(current_shirt_name)
        
        if confirmed:
            # Delete the file
            if self.upload_manager.delete_shirt(current_shirt_name):
                print(f"Successfully deleted: {current_shirt_name}")
                
                # Reload shirts
                self.virtual_try_on.shirt_images = self.virtual_try_on.load_shirt_images()
                
                # Adjust index if needed
                if self.virtual_try_on.shirt_images:
                    if current_index >= len(self.virtual_try_on.shirt_images):
                        self.virtual_try_on.current_shirt_index = len(self.virtual_try_on.shirt_images) - 1
                    else:
                        self.virtual_try_on.current_shirt_index = current_index
                    
                    print(f"Now showing: {self.virtual_try_on.get_current_shirt_name()}")
                else:
                    self.virtual_try_on.current_shirt_index = -1
                    print("No shirts remaining")
            else:
                print(f"Failed to delete: {current_shirt_name}")
        else:
            print("Deletion cancelled")

    def show_delete_confirmation_dialog(self, shirt_name):
        title = "Delete Confirmation"
        message = f"Are you sure you want to delete '{shirt_name}'? This action cannot be undone."
        
        # Reset dialog state
        self.confirmation_dialog.confirmed = None
        self.confirmation_dialog.is_showing = True
        
        # Pause main processing
        was_running = self.running
        self.running = False
        
        print("Confirmation dialog opened - waiting for response...")
        
        # Dialog loop
        while self.confirmation_dialog.confirmed is None and self.confirmation_dialog.is_showing:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Continue with normal rendering
            landmarks, pose_results = self.pose_detector.detect_pose(frame)
            
            if landmarks:
                key_points = self.pose_detector.get_key_points(landmarks)
                
                if key_points is None:
                    key_points = {}
                
                # Apply shirt overlay
                if key_points and self.virtual_try_on.shirt_images and self.show_shirt_overlay:
                    try:
                        current_shirt = self.virtual_try_on.shirt_images[
                            self.virtual_try_on.current_shirt_index
                        ][1]
                        
                        if self.use_gan and self.gan_tryon:
                            result = self.gan_tryon.style_transfer(frame, current_shirt, key_points)
                            if result is not None:
                                frame = result
                        else:
                            result = self.virtual_try_on.overlay_shirt(frame, key_points)
                            if result is not None:
                                frame = result
                    except:
                        pass
            
            # Draw confirmation dialog on top
            frame = self.confirmation_dialog.show(frame, title, message)
            
            cv2.imshow(self.window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(10) & 0xFF
            
            # Check keyboard response
            keyboard_result = self.confirmation_dialog.handle_keypress(key)
            if keyboard_result is not None:
                self.confirmation_dialog.confirmed = keyboard_result
        
        # Get the final result
        confirmed = self.confirmation_dialog.confirmed
        
        # Reset dialog
        self.confirmation_dialog.is_showing = False
        self.confirmation_dialog.confirmed = None
        
        # Resume main processing
        self.running = was_running
        
        print(f"Dialog result: {'CONFIRMED' if confirmed else 'CANCELLED'}")
        
        return confirmed if confirmed is not None else False


    def draw_shirt_carousel(self, frame):
        if not self.virtual_try_on.shirt_images:
            return frame
        
        total_shirts = len(self.virtual_try_on.shirt_images)
        current_idx = self.virtual_try_on.current_shirt_index
        
        # Carousel dimensions
        thumbnail_width = 120
        thumbnail_height = 120
        spacing = 30
        bottom_margin = 20

        # Carousel positions
        frame_height, frame_width = frame.shape[:2]
        carousel_y = frame_height - thumbnail_height - bottom_margin - 40
        center_x = frame_width // 2
        
        # Background for carousel
        overlay = frame.copy()
        carousel_bg_height = thumbnail_height + 100
        cv2.rectangle(overlay, 
                    (0, frame_height - carousel_bg_height),
                    (frame_width, frame_height),
                    (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Indices for previous, current, next
        prev_idx = (current_idx - 1) % total_shirts
        next_idx = (current_idx + 1) % total_shirts
        
        # Previous first, Next second, Current last
        shirts_to_display = [
            (prev_idx, center_x - thumbnail_width - spacing, "Previous", (150, 150, 150), False),
            (next_idx, center_x + spacing, "Next", (150, 150, 150), False),
            (current_idx, center_x - thumbnail_width // 2, "Current", (0, 255, 0), True)
        ]
        
        for idx, x_pos, label, border_color, is_current in shirts_to_display:
            shirt_name, shirt_img = self.virtual_try_on.shirt_images[idx]
            
            try:
                # Create white background
                thumbnail = np.ones((thumbnail_height, thumbnail_width, 3), dtype=np.uint8) * 255
                
                # Get original shirt dimensions
                orig_h, orig_w = shirt_img.shape[:2]
                
                # Calculate scale to fit within thumbnail while maintaining aspect ratio
                scale = min(thumbnail_width / orig_w, thumbnail_height / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                
                # Resize the shirt
                resized_shirt = cv2.resize(shirt_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Calculate centering offset
                y_offset = (thumbnail_height - new_h) // 2
                x_offset = (thumbnail_width - new_w) // 2
                
                # Check if image has alpha channel (BGRA format)
                if resized_shirt.shape[2] == 4:
                    # Extract BGR and alpha channels
                    bgr = resized_shirt[:, :, :3]
                    alpha = resized_shirt[:, :, 3] / 255.0
                    
                    # Blend with white background using alpha
                    for c in range(3):
                        thumbnail[y_offset:y_offset+new_h, 
                                x_offset:x_offset+new_w, c] = \
                            (alpha * bgr[:, :, c] + 
                            (1 - alpha) * thumbnail[y_offset:y_offset+new_h, 
                                                    x_offset:x_offset+new_w, c]).astype(np.uint8)
                else:
                    # Direct copy for BGR images
                    thumbnail[y_offset:y_offset+new_h, 
                            x_offset:x_offset+new_w] = resized_shirt
                
                # Place thumbnail on frame
                y_start = carousel_y
                y_end = y_start + thumbnail_height
                x_start = x_pos
                x_end = x_start + thumbnail_width
                
                # Ensure coordinates are within frame bounds
                if x_start >= 0 and x_end <= frame_width and y_start >= 0 and y_end <= frame_height:
                    frame[y_start:y_end, x_start:x_end] = thumbnail
                    
                    # Draw border
                    thickness = 5 if is_current else 2
                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), border_color, thickness)
                    
                    # Draw label
                    label_y = y_start - 10
                    font_scale = 0.6 if is_current else 0.5
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                    
                    if label == "Previous":
                        label_x = x_start
                    elif label == "Current":
                        label_x = x_start + (thumbnail_width - label_size[0]) // 2
                    elif label == "Next":
                        label_x = x_end - label_size[0]
                    else:
                        label_x = x_start
                    
                    cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, border_color, 2)
                    
                    # Draw shirt name below thumbnail for current shirt
                    if is_current:
                        display_name = shirt_name[:15] + "..." if len(shirt_name) > 15 else shirt_name
                        text_size = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        text_x = x_start + (thumbnail_width - text_size[0]) // 2
                        text_y = y_end + 20
                        cv2.putText(frame, display_name, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            except Exception as e:
                print(f"Error drawing thumbnail for {shirt_name}: {e}")
                continue
        
        # Draw navigation hints at the very bottom
        hint_y = frame_height - 10
        left_hint = "< LEFT HAND"
        right_hint = "RIGHT HAND >"
        
        cv2.putText(frame, left_hint, (20, hint_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        right_hint_size = cv2.getTextSize(right_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, right_hint, (frame_width - right_hint_size[0] - 20, hint_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame
    
    def draw_ui(self, frame):
        # Current shirt name
        shirt_name = self.virtual_try_on.get_current_shirt_name()
        cv2.putText(frame, f"Current: {shirt_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Shirts count
        total_shirts = len(self.virtual_try_on.shirt_images)
        current_index = self.virtual_try_on.current_shirt_index + 1 if total_shirts > 0 else 0
        cv2.putText(frame, f"Shirt {current_index}/{total_shirts}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # GAN mode indicator
        if self.use_gan and self.gan_tryon:
            cv2.putText(frame, "GAN MODE: ON", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Keypoints status indicator
        keypoint_status = "ON" if self.show_keypoints else "OFF"
        keypoint_color = (0, 255, 0) if self.show_keypoints else (100, 100, 100)
        cv2.putText(frame, f"Keypoints: {keypoint_status}", (10, 145), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, keypoint_color, 2)
        
        # Shirt overlay status indicator
        overlay_status = "ON" if self.show_shirt_overlay else "OFF"
        overlay_color = (0, 255, 0) if self.show_shirt_overlay else (255, 0, 0)
        cv2.putText(frame, f"Shirt Overlay: {overlay_status}", (10, 175), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_color, 2)
        
        instructions = [
            "Controls:",
            "U - Upload | D - Delete",
            "G - Toggle GAN",
            "K - Keypoints | V - Overlay",
            "N/P - Next/Prev",
            "S - Screenshot",
            "Q - Quit",
            "",
            "Click dropdown to select"
        ]
        
        x_pos = 10
        y_pos = 220
        
        for i, instruction in enumerate(instructions):
            if i == 1:
                color = (0, 255, 255)
                thickness = 2
            elif i == 3:
                color = (255, 255, 0)
                thickness = 2
            elif i == 8:
                color = (255, 200, 0)
                thickness = 2
            else:
                color = (255, 255, 255)
                thickness = 1
            
            cv2.putText(frame, instruction, (x_pos, y_pos + i*25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness)
        
        # Shirt carousel at bottom
        frame = self.draw_shirt_carousel(frame)
        
        # Dropdown menu LAST
        if self.virtual_try_on.shirt_images:
            shirt_names = [name for name, _ in self.virtual_try_on.shirt_images]
            frame = self.dropdown.draw(
                frame,
                shirt_names,
                self.virtual_try_on.current_shirt_index
            )
        
        return frame
 
    def mouse_callback(self, event, x, y, flags, param):
        if self.confirmation_dialog.is_showing:
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Mouse clicked at ({x}, {y}) while dialog is showing")
                result = self.confirmation_dialog.handle_click(x, y)
                if result is not None:
                    print(f"   Dialog responded with: {result}")
                else:
                    print(f"   Click was not on any button")
            return
        
        # Handle dropdown
        if not self.virtual_try_on.shirt_images:
            return
        
        shirt_names = [name for name, _ in self.virtual_try_on.shirt_images]
        
        if event == cv2.EVENT_MOUSEMOVE:
            self.dropdown.handle_hover(x, y, len(shirt_names))
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            new_idx = self.dropdown.handle_click(
                x, y,
                len(shirt_names),
                self.virtual_try_on.current_shirt_index
            )
            
            if new_idx is not None:
                self.virtual_try_on.current_shirt_index = new_idx
                print(f"Selected: {shirt_names[new_idx]}")
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            self.dropdown.handle_scroll(-delta, len(shirt_names))



    def run(self):
        try:
            while self.cap.isOpened() and self.running:
                if self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame from camera")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    self.current_frame = frame.copy()
                    
                    # Detect pose using YOLOv8
                    landmarks, pose_results = self.pose_detector.detect_pose(frame)
                    
                    if landmarks:
                        key_points = self.pose_detector.get_key_points(landmarks)
                        
                        # Validate keypoints
                        if key_points is None:
                            key_points = {}
                        
                        # Hand raise detection)
                        left_up, right_up = self.detect_hand_raised(frame, key_points)
                        self.handle_hand_gestures(left_up, right_up)
                        
                        # Apply shirt overlay if enabled
                        if key_points and self.virtual_try_on.shirt_images and self.show_shirt_overlay:
                            current_shirt = self.virtual_try_on.shirt_images[
                                self.virtual_try_on.current_shirt_index
                            ][1]
                            
                            # Apply GAN or standard overlay
                            if self.use_gan and self.gan_tryon:
                                frame = self.gan_tryon.style_transfer(
                                    frame, current_shirt, key_points
                                )
                            else:
                                frame = self.virtual_try_on.overlay_shirt(frame, key_points)
                        
                        # Draw keypoints if enabled
                        if self.show_keypoints and key_points:
                            frame = self.draw_keypoints(frame, landmarks, key_points)
                    
                    # Draw UI elements (includes carousel and dropdown)
                    frame = self.draw_ui(frame)
                    
                    # Store the frame WITH the shirt overlay for screenshots
                    self.display_frame = frame.copy()
                    
                    # Display the frame
                    cv2.imshow(self.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_keypress(key)
                    
                    if not self.running:
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"\nError during execution: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\nCleaning up")
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            if hasattr(self, 'hands') and self.hands is not None:
                self.hands.close()
            cv2.destroyAllWindows()
            print("Virtual Try-On System Stopped!")




if __name__ == "__main__":
    try:
        app = VirtualTryOnApp()
        app.run()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\nExiting...")
