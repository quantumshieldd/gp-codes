import os
import numpy as np

base_path = "G:/My Drive/"
clean_path = f"{base_path}/gp2/processed_images"
attack_paths = {
    'hsja': f"{base_path}/attacks/hsja_success_npy",
    'pgd': f"{base_path}/attacks/PGD",
    'fgsm': f"{base_path}/attacks/FGSM",
    'boundary': f"{base_path}/attacks/boundary_attacked_npy",
    'patch': f"{base_path}/attacks/Patch_Attack",
    'color': f"{base_path}/attacks/Color_Space_Attack",
    'weather_rain': f"{base_path}/attacks/Weather_Attack/Rain_Attack",
    'weather_fog': f"{base_path}/attacks/Weather_Attack/Fog_Attack",
    'gaussian': f"{base_path}/attacks/gussian_images_attack",
}

#Label mapping
label_map = {
    'clean': 0,
    'attack': 1
}

#Loader for clean/attacked .npy files

def load_npy_images(folder, label, max_count=None):
    images, labels = [], []
    count = 0
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".npy"):
            continue
        path = os.path.join(folder, file)
        try:
            data = np.load(path, allow_pickle=True)

            #Handle scalar-wrapped dicts
            if isinstance(data, np.ndarray) and data.shape == ():
                data = data.item()

            if isinstance(data, dict) and 'image' in data:
                img = data['image']
            else:
                img = data

            if not isinstance(img, np.ndarray):
                raise ValueError("Image data is not a NumPy array")

            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"Invalid image dimensions: {img.shape}")

            # Convert to float32
            img = img.astype(np.float32)

            # Check if image is already normalized to [0, 1]
            img_max = img.max()
            if img_max <= 1.0:
                # Scale back to [0, 255]
                img = img * 255.0

            # Clip pixel values to [0, 255]
            img = np.clip(img, 0, 255)

            images.append(img)
            labels.append(label)
            count += 1

            if max_count and count >= max_count:
                break

            if count % 100 == 0:
                print(f"Loaded {count} from {folder}...")

        except Exception as e:
            print(f" Skipping {file} due to error: {e}")

    print(f" Finished loading {count} images from {folder}")
    return images, labels

#Load clean + attacked
print("\n Loading clean images (limit 5000)...")
X, y = load_npy_images(clean_path, label_map['clean'], max_count=5000)

for attack, path in attack_paths.items():
    print(f"\n Loading {attack} (limit 500)...")
    imgs, labels = load_npy_images(path, label_map['attack'], max_count=500)
    X.extend(imgs)
    y.extend(labels)

#Convert and Save
X = np.array(X)
y = np.array(y)

#Verify pixel value range
print(f"\n📏 Pixel value range: Min={X.min():.2f}, Max={X.max():.2f}")
print(f"Total images loaded: {len(X)}")
print(f"Example image shape: {X[0].shape if len(X) > 0 else 'N/A'}")
print(f"Labels shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

np.save(f"{base_path}/X1_preprocessed.npy", X)
np.save(f"{base_path}/y1_labels.npy", y)
print("\n Saved X1_preprocessed.npy and y1_labels.npy")