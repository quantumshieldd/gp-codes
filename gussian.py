import os
import numpy as np
import random
import cv2

# Paths
input_dir = r'C:\Users\ASUS\Desktop\gp\processed_images'
output_dir = r'C:\Users\ASUS\Desktop\gp\gussian_images_attack'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load all .npy files with at least one annotation
npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
valid_files = []

for file in npy_files:
    data = np.load(os.path.join(input_dir, file), allow_pickle=True).item()
    if len(data['annotations']) > 0:
        valid_files.append(file)

print(f"Found {len(valid_files)} files with annotations.")

# Select 1000 random files
selected_files = random.sample(valid_files, min(1000, len(valid_files)))

# Apply Gaussian Blur
for file in selected_files:
    file_path = os.path.join(input_dir, file)
    data = np.load(file_path, allow_pickle=True).item()
    
    image = data['image']
    annotations = data['annotations']

    # Convert image back to uint8 for OpenCV
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image_uint8, (11, 11), 5)
    
    # Normalize again and save
    blurred = blurred.astype(np.float32) / 255.0
    save_data = {'image': blurred, 'annotations': annotations}
    
    np.save(os.path.join(output_dir, file), save_data)
    print(f"📦 Saved blurred: {file}")

print(" Gaussian attack applied to 1000 images.")
