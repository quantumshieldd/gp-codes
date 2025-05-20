import numpy as np
import cv2
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the model 
model_path = '/content/drive/MyDrive/GP/kitti_multilabel_cnn.h5'
model = tf.keras.models.load_model(model_path)

# Define paths for patch attack
image_dir = '/content/drive/MyDrive/GP/processed_images'
patch_output_dir = '/content/drive/MyDrive/GP/Patch_Attack'
os.makedirs(patch_output_dir, exist_ok=True)

# Load class labels from a file 
class_labels = '/content/drive/MyDrive/GP/class_names.npy'  

# Function to create a simple patch 
def create_patch(size=(32, 32), color=(255, 0, 0)):
    patch = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    patch[:, :] = color  # Color the entire patch
    return patch

# Function to apply the patch to a random position on the image
def apply_patch(image, patch):
    h, w, _ = image.shape
    ph, pw, _ = patch.shape
    
    # Randomly choose a position for the patch
    x = random.randint(0, w - pw)
    y = random.randint(0, h - ph)
    
    # Apply the patch to the image
    image_with_patch = image.copy()
    image_with_patch[y:y+ph, x:x+pw] = patch
    
    return image_with_patch

# Function to apply the patch attack and check the label change
def test_patch_attack(image_dir, patch_output_dir, model, class_labels, max_images=1000, batch_size=10):
    success_count = 0
    processed_images = set()  # Set to track processed images

    # Load processed images from previous runs to avoid duplication
    if os.path.exists('processed_images.txt'):
        with open('processed_images.txt', 'r') as f:
            processed_images = set(f.read().splitlines())

    image_files = sorted(os.listdir(image_dir))
    random.shuffle(image_files)

    for fname in image_files[:max_images]:
        if not fname.endswith('.npy') or fname in processed_images:
            continue

        # Load image
        image_data = np.load(os.path.join(image_dir, fname), allow_pickle=True).item()
        original_image = image_data['image']

        # Get the model's prediction before the attack (convert prediction to label)
        original_image_resized = cv2.resize(original_image, (512, 256))
        original_prediction = model.predict(np.expand_dims(original_image_resized, axis=0))

        # For multi-label prediction, we check the threshold or get the top class
        if len(original_prediction.shape) > 1:  # Multi-label output
            original_labels = [class_labels[i] for i in range(len(original_prediction[0])) if original_prediction[0][i] > 0.5]
            original_label = ', '.join(original_labels)  # Multiple labels may apply
        else:  # Single class output (softmax or sigmoid for multi-class)
            original_label = class_labels[np.argmax(original_prediction)]

        # Apply patch attack
        patch = create_patch(size=(32, 32), color=(255, 0, 0))  
        adversarial_image = apply_patch(original_image, patch)

        # Get the model's prediction after the attack
        adversarial_image_resized = cv2.resize(adversarial_image, (512, 256))
        adversarial_prediction = model.predict(np.expand_dims(adversarial_image_resized, axis=0))

        # For multi-label prediction, we check the threshold or get the top class
        if len(adversarial_prediction.shape) > 1:  # Multi-label output
            adversarial_labels = [class_labels[i] for i in range(len(adversarial_prediction[0])) if adversarial_prediction[0][i] > 0.5]
            adversarial_label = ', '.join(adversarial_labels)  # Multiple labels may apply
        else:  # Single class output
            adversarial_label = class_labels[np.argmax(adversarial_prediction)]

        # Compare predictions before and after the attack
        print(f"Original Label: {original_label}")
        print(f"Adversarial Label: {adversarial_label}")
        
        if original_label != adversarial_label:  # Check if the label has changed
            # Save the adversarial image only if the attack was successful (label change)
            output_path = os.path.join(patch_output_dir, f'{fname.split(".")[0]}.npy')
            np.save(output_path, {'image': adversarial_image, 'annotations': image_data['annotations']})
            print(f"Adversarial patch image saved to {output_path}")
            success_count += 1

            # Display the original and adversarial images side by side
            plt.figure(figsize=(10,5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Original Image\nLabel: {original_label}")
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(adversarial_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Adversarial Image\nLabel: {adversarial_label}")
            plt.show()

            # Track the processed image to avoid reprocessing
            processed_images.add(fname)

            # Save the processed image list to file for future runs
            with open('processed_images.txt', 'a') as f:
                f.write(f"{fname}\n")

    return success_count

# Test the patch attack on images
success_count = test_patch_attack(image_dir, patch_output_dir, model, class_labels, max_images=1000, batch_size=10)

print(f"Total successful patch attacks: {success_count}")
