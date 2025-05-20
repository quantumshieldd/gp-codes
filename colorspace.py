import numpy as np
import cv2
import os
import tensorflow as tf
import random

# Function to apply Color Space Attack (Hue, Saturation, Lightness modification)
def strong_color_space_attack(image, hue_shift=50, saturation_factor=2.0, lightness_factor=1.5):
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Stronger hue shift (larger color changes)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_shift) % 180  # Hue in the range [0, 180]
    
    # Stronger saturation change (more vibrant or dull)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_factor, 0, 255)

    # Stronger lightness (making it significantly brighter or darker)
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] * lightness_factor, 0, 255)

    # Convert back to RGB
    adversarial_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return adversarial_image

# Function to apply Color Space Attack and resize the adversarial image
def apply_color_space_attack_to_images(image_dir, output_dir, model, max_images=1000):
    # Get all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]
    
    # Load the list of processed images (if it exists)
    processed_images_file = os.path.join(output_dir, "processed_images.npy")
    if os.path.exists(processed_images_file):
        processed_images = list(np.load(processed_images_file))
    else:
        processed_images = []

    # Filter out images that have already been processed
    image_files = [f for f in image_files if f not in processed_images]
    
    # Shuffle and select a sample of 1000 images (if there are less than 1000, process all remaining)
    random.shuffle(image_files)
    image_files = image_files[:max_images]

    success_count = 0
    processed_images = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for fname in image_files:
        image_path = os.path.join(image_dir, fname)
        output_path = os.path.join(output_dir, fname)

        # Load the original image
        data = np.load(image_path, allow_pickle=True)
        original_image = data.item().get('image', None)

        if original_image is None:
            print(f"Error: Image data not found for {image_path}.")
            continue

        # Apply stronger Color Space Attack
        adversarial_image = strong_color_space_attack(original_image)

        # Resize the adversarial image to match the original image size
        adversarial_image_resized = cv2.resize(adversarial_image, (original_image.shape[1], original_image.shape[0]))

        # Save the adversarial image only for successful attacks
        np.save(output_path, {'image': adversarial_image_resized})
        print(f"Adversarial image saved to {output_path}")

        # Increment the success count
        success_count += 1
        processed_images.append(fname)

    # Save the updated list of processed images
    np.save(processed_images_file, np.array(processed_images))

    # Print the total successful attacks
    print(f"Total successful adversarial attacks applied: {success_count}")

# Load your model
model_path = '/content/drive/MyDrive/GP/kitti_multilabel_cnn.h5'
model = tf.keras.models.load_model(model_path)

# Define paths
image_dir = '/content/drive/MyDrive/GP/processed_images'
output_dir = '/content/drive/MyDrive/GP/Color_Space_Attack'

# Apply the attack and save only the successful adversarial images
apply_color_space_attack_to_images(image_dir, output_dir, model)
