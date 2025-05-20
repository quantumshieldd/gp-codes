import numpy as np
import cv2
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Define paths for rain attack
rain_output_dir = '/content/drive/MyDrive/gp2/Rain_Attack'
image_dir = '/content/drive/MyDrive/gp2/processed_images'

# Define the path to your trained model
model_path = '/content/drive/MyDrive/gp2/kitti_multilabel_cnn.h5'
model = tf.keras.models.load_model(model_path)  # Load the model
success_count_rain = 0
processed_images_rain = set()

# Ensure the output folder exists
os.makedirs(rain_output_dir, exist_ok=True)

# Function to simulate rain by adding random lines (raindrops)
def apply_rain(image, intensity=50):
    rain_image = image.copy()
    height, width, _ = rain_image.shape
    for _ in range(intensity):
        # Generate random raindrop lines
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        color = np.random.randint(0, 255, size=3)
        thickness = np.random.randint(1, 3)
        cv2.line(rain_image, (x1, y1), (x2, y2), color.tolist(), thickness)
    return rain_image

# Function to apply the rain attack and save successful images
def apply_rain_attack_to_images(image_dir, rain_output_dir, model, max_images=500, batch_size=10):
    global success_count_rain
    image_files = sorted(os.listdir(image_dir))
    random.shuffle(image_files)

    # Process images in batches
    for batch_start in range(0, min(max_images, len(image_files)), batch_size):
        batch_files = image_files[batch_start:batch_start + batch_size]
        for fname in batch_files:
            if not fname.endswith('.npy') or fname in processed_images_rain:
                continue

            # Load image
            image_data = np.load(os.path.join(image_dir, fname), allow_pickle=True).item()
            original_image = image_data['image']

            # Apply rain attack
            adversarial_image = apply_rain(original_image)

            # Resize both original and adversarial images to match model input shape
            original_image_resized = cv2.resize(original_image, (512, 256))
            adversarial_image_resized = cv2.resize(adversarial_image, (512, 256))

            # Run model inference on original and adversarial images
            original_prediction = model.predict(np.expand_dims(original_image_resized, axis=0))
            adversarial_prediction = model.predict(np.expand_dims(adversarial_image_resized, axis=0))

            # Check if the attack was successful (prediction difference)
            if not np.array_equal(original_prediction, adversarial_prediction):
                # Save the adversarial image only if the attack was successful
                output_path = os.path.join(rain_output_dir, f'{fname.split(".")[0]}.npy')
                np.save(output_path, {'image': adversarial_image, 'annotations': image_data['annotations']})
                print(f" Adversarial rain image saved to {output_path}")
                success_count_rain += 1
                processed_images_rain.add(fname)

    return success_count_rain

# Apply rain attack to the images and save the results
success_count_rain = apply_rain_attack_to_images(image_dir, rain_output_dir, model, max_images=500, batch_size=10)

print(f"Total successful rain attacks applied: {success_count_rain}")
