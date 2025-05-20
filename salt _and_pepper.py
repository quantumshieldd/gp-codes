import os
import random
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

#Paths 
data_dir = r"C:\Users\ASUS\Desktop\gp2\processed_images"
model_path = r"C:\Users\ASUS\Desktop\gp\kitti_multilabel_cnn.h5"
class_names_path = r"G:\My Drive\gp2\class_names.npy"
output_dir = r"C:\Users\ASUS\Desktop\gp\salt_pepper"
os.makedirs(output_dir, exist_ok=True)

#Load model and class names 
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Falling back to rebuilding model architecture...")
    # Rebuild model architecture from CNN.py
    class_names = np.load(class_names_path, allow_pickle=True)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 512, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='sigmoid')
    ])
    # Load weights
    model.load_weights(model_path)
    print("Rebuilt model and loaded weights")

#Load class names
class_names = np.load(class_names_path, allow_pickle=True)

#Salt-and-Pepper Noise Function
def add_salt_pepper_noise(image, noise_density=0.05):
    """
    Add salt-and-pepper noise to an image.
    noise_density: Fraction of pixels to corrupt (0 to 1).
    """
    img = image.copy()
    h, w, c = img.shape
    n_pixels = h * w
    n_noise = int(noise_density * n_pixels)
    
    # Randomly select pixels for salt (white) and pepper (black)
    coords = [np.random.randint(0, dim, n_noise) for dim in [h, w]]
    
    # Add salt (white: 1.0)
    img[coords[0][:n_noise//2], coords[1][:n_noise//2], :] = 1.0
    
    # Add pepper (black: 0.0)
    img[coords[0][n_noise//2:], coords[1][n_noise//2:], :] = 0.0
    
    return img

npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
if len(npy_files) < 50:
    raise ValueError(f"Only {len(npy_files)} .npy files found, need at least 50.")

# --- Process images until 50 successful attacks ---
successful_attacks = 0
processed_files = set()  # Track processed images
available_files = npy_files.copy()  # Copy to avoid modifying original list
total_processed = 0

while successful_attacks < 50 and available_files:
    # Randomly select an image
    sample_file = random.choice(available_files)
    available_files.remove(sample_file) 
    processed_files.add(sample_file)
    total_processed += 1

    # Load sample image
    file_path = os.path.join(data_dir, sample_file)
    data = np.load(file_path, allow_pickle=True).item()
    clean_image = data["image"]  # Shape: (256, 512, 3), normalized [0, 1]
    clean_labels = [ann["class"] for ann in data["annotations"] if ann["class"] != "DontCare"]

    # Generate adversarial image
    adv_image = add_salt_pepper_noise(clean_image, noise_density=0.05)

    # Predict on clean and adversarial images
    clean_pred = model.predict(clean_image[np.newaxis, ...], verbose=0)[0]
    adv_pred = model.predict(adv_image[np.newaxis, ...], verbose=0)[0]

    # Decode predictions
    threshold = 0.5
    clean_classes = class_names[clean_pred > threshold].tolist()
    adv_classes = class_names[adv_pred > threshold].tolist()

    # Check if attack was successful (predictions differ)
    if clean_classes != adv_classes:
        successful_attacks += 1

        # Save adversarial image (.npy)
        output_npy_path = os.path.join(output_dir, f"adv_salt_pepper_{sample_file}")
        np.save(output_npy_path, {"image": adv_image, "annotations": data["annotations"]})
        print(f" Adversarial .npy saved: {output_npy_path}")

        # Save adversarial image (.png)
        adv_image_png = (adv_image * 255).astype(np.uint8)
        output_png_path = os.path.join(output_dir, f"adv_salt_pepper_{sample_file.replace('.npy', '.png')}")
        cv2.imwrite(output_png_path, cv2.cvtColor(adv_image_png, cv2.COLOR_RGB2BGR))
        print(f" Adversarial .png saved: {output_png_path}")

        # Create side-by-side visualization with annotations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Clean image
        ax1.imshow(clean_image)
        ax1.set_title(f"Clean: {sample_file}", fontsize=10)
        ax1.text(0, -20, f"Predicted: {', '.join(clean_classes) if clean_classes else 'None'}\nGround-Truth: {', '.join(clean_labels) if clean_labels else 'None'}", 
                 fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.8))
        ax1.axis("off")
        
        # Adversarial image
        ax2.imshow(adv_image)
        ax2.set_title(f"Attacked: adv_salt_pepper_{sample_file}", fontsize=10)
        ax2.text(0, -20, f"Predicted: {', '.join(adv_classes) if adv_classes else 'None'}\nGround-Truth: {', '.join(clean_labels) if clean_labels else 'None'}", 
                 fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.8))
        ax2.axis("off")
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"vis_salt_pepper_{sample_file.replace('.npy', '.png')}")
        plt.savefig(vis_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f" Annotated visualization saved: {vis_path}")
    else:
        print(f" Attack failed for {sample_file}: Predictions unchanged")

    # Progress update
    print(f"Progress: {successful_attacks}/50 successful attacks, {total_processed} images processed")

if successful_attacks == 50:
    print(f" Generated exactly 50 successful adversarial images with annotated visualizations!")
else:
    print(f"Stopped: Only {successful_attacks} successful attacks achieved after processing {total_processed} images. Check noise_density or model robustness.")