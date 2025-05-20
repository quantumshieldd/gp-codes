import numpy as np
import cv2
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Define paths for fog attack
image_dir = '/content/drive/MyDrive/GP/processed_images'
fog_output_dir = '/content/drive/MyDrive/GP/Fog_Attack'
success_count_fog = 0
processed_images = set()

os.makedirs(fog_output_dir, exist_ok=True)

# Load the model 
model_path = '/content/drive/MyDrive/GP/kitti_multilabel_cnn.h5'
model = tf.keras.models.load_model(model_path)

# Function to simulate fog by applying Gaussian blur
def apply_fog(image, intensity=15):
    # Ensure intensity is an odd number
    if intensity % 2 == 0:
        intensity += 1
    return cv2.GaussianBlur(image, (intensity, intensity), 0)

# Function to apply the fog attack and save successful images
def apply_fog_attack_to_images(image_dir, fog_output_dir, model, max_images=500, batch_size=10):
    global success_count_fog
    image_files = sorted(os.listdir(image_dir))
    random.shuffle(image_files)
    
    # Process images in batches
    for batch_start in range(0, min(max_images, len(image_files)), batch_size):
        batch_files = image_files[batch_start:batch_start + batch_size]
        for fname in batch_files:
            if not fname.endswith('.npy') or fname in processed_images:
                continue

            # Load image
            image_data = np.load(os.path.join(image_dir, fname), allow_pickle=True).item()
            original_image = image_data['image']

            # Apply fog attack
            adversarial_image = apply_fog(original_image)

            # Resize both original and adversarial images to match model input shape
            original_image_resized = cv2.resize(original_image, (512, 256))
            adversarial_image_resized = cv2.resize(adversarial_image, (512, 256))

            # Run model inference on original and adversarial images
            original_prediction = model.predict(np.expand_dims(original_image_resized, axis=0))
            adversarial_prediction = model.predict(np.expand_dims(adversarial_image_resized, axis=0))

            # Check if the attack was successful (prediction difference)
            if not np.array_equal(original_prediction, adversarial_prediction):
                # Save the adversarial image only if the attack was successful
                output_path = os.path.join(fog_output_dir, f'{fname.split(".")[0]}.npy')
                np.save(output_path, {'image': adversarial_image, 'annotations': image_data['annotations']})
                print(f"Adversarial fog image saved to {output_path}")
                success_count_fog += 1
                processed_images.add(fname)

    return success_count_fog

# Apply fog attack to the images and save the results
success_count_fog = apply_fog_attack_to_images(image_dir, fog_output_dir, model, max_images=500, batch_size=10)

print(f"Total successful fog attacks applied: {success_count_fog}")
