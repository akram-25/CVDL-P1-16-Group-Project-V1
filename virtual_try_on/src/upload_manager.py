import os
from tkinter import Tk, filedialog
from PIL import Image
from rembg import remove

class ShirtUploadManager:
    def __init__(self, shirts_folder="../assets/shirts/"):
        self.shirts_folder = shirts_folder
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']

        shirts_path = os.path.join(os.path.dirname(__file__), self.shirts_folder)
        os.makedirs(shirts_path, exist_ok=True)
        
        print("Rembg background removal enabled")

    def open_file_dialog(self):
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        filetypes = [
            ('Image Files', '*.png *.jpg *.jpeg *.bmp *.webp'),
            ('PNG Files', '*.png'),
            ('JPEG Files', '*.jpg *.jpeg'),
            ('All Files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select a Shirt Image",
            filetypes=filetypes
        )
        
        root.destroy()
        return file_path if file_path else None

    def remove_background_rembg(self, image_path):
        try:
            print("Removing background with rembg...")
            
            # Read the input image
            with open(image_path, 'rb') as input_file:
                input_data = input_file.read()
            
            # Remove background
            output_data = remove(input_data)
            
            # Convert to PIL Image
            from io import BytesIO
            output_img = Image.open(BytesIO(output_data))
            
            print("Background removed successfully")
            return output_img
            
        except Exception as e:
            print(f"Error removing background: {e}")
            return None

    def process_and_save_shirt(self, source_path, custom_name=None):
        try:
            # Remove background using rembg
            processed_img = self.remove_background_rembg(source_path)
            
            if processed_img is None:
                print("Background removal failed, saving original image")
                processed_img = Image.open(source_path)

            # Ensure RGBA mode for transparency
            if processed_img.mode != 'RGBA':
                processed_img = processed_img.convert('RGBA')

            # Generate filename
            if custom_name:
                filename = f"{custom_name}.png"
            else:
                original_name = os.path.splitext(os.path.basename(source_path))[0]
                filename = f"{original_name}.png"

            shirts_path = os.path.join(os.path.dirname(__file__), self.shirts_folder)
            save_path = os.path.join(shirts_path, filename)

            # Handle duplicate filenames
            counter = 1
            base_name = os.path.splitext(filename)[0]
            while os.path.exists(save_path):
                filename = f"{base_name}_{counter}.png"
                save_path = os.path.join(shirts_path, filename)
                counter += 1

            # Resize if too large
            max_size = 2048
            if processed_img.width > max_size or processed_img.height > max_size:
                processed_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Save as PNG
            processed_img.save(save_path, 'PNG')
            print(f"Shirt saved permanently: {filename}")
            return save_path, filename

        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None

    def upload_shirt(self):
        file_path = self.open_file_dialog()

        if not file_path:
            print("No file selected")
            return None, None

        print(f"Selected file: {file_path}")
        return self.process_and_save_shirt(file_path)
    
    def delete_shirt(self, filename):
        try:
            shirts_path = os.path.join(os.path.dirname(__file__), self.shirts_folder)
            file_path = os.path.join(shirts_path, filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {filename}")
                return True
            else:
                print(f"File not found: {filename}")
                return False
                
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    def get_shirt_path(self, filename):
        shirts_path = os.path.join(os.path.dirname(__file__), self.shirts_folder)
        return os.path.join(shirts_path, filename)