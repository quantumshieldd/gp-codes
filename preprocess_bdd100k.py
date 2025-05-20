# BDD100K Preprocessing Script
# Goal: Load images + labels, resize & normalize, attach annotations, and save as .npy files for attacks

import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# === CONFIG ===
IMAGES_DIR = r"C:\Users\eslam\Desktop\archive\bdd100k\bdd100k\images\100k\train\trainA"
LABELS_PATH = r"C:\Users\eslam\Desktop\archive\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"
OUTPUT_DIR = r"C:\Users\eslam\Desktop\bdd100k_preprocessed"
TARGET_SHAPE = (256, 512)  # (height, width)

# === CLASSES TO INCLUDE (customize this list as needed) ===
VALID_CLASSES = {"car", "truck", "bus", "person", "traffic light", "traffic sign"}

# === Create output folder ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load annotations ===
with open(LABELS_PATH, 'r') as f:
    annotations = json.load(f)

# === Build dictionary from name to annotation ===
ann_dict = {item['name']: item for item in annotations}

# === Process each image ===
image_filenames = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg') and f in ann_dict]

for filename in tqdm(image_filenames[:10], desc="Preprocessing BDD100K"):
    image_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Skipping unreadable file: {filename}")
        continue

    # Resize & normalize image
    img_resized = cv2.resize(img, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Extract annotations for valid objects only
    objects = ann_dict[filename].get('labels', [])
    filtered_ann = []

    orig_h, orig_w = img.shape[:2]
    scale_x = TARGET_SHAPE[1] / orig_w
    scale_y = TARGET_SHAPE[0] / orig_h

    for obj in objects:
        if obj['category'] in VALID_CLASSES and 'box2d' in obj:
            box = obj['box2d']
            x1 = int(box['x1'] * scale_x)
            y1 = int(box['y1'] * scale_y)
            x2 = int(box['x2'] * scale_x)
            y2 = int(box['y2'] * scale_y)

            filtered_ann.append({
                "category": obj['category'],
                "bbox": [x1, y1, x2, y2]
            })

    # Save as .npy file
    output_data = {
        "image": img_normalized,
        "annotations": filtered_ann
    }
    out_path = os.path.join(OUTPUT_DIR, filename.replace('.jpg', '.npy'))
    np.save(out_path, output_data)

print("Preprocessing completed for first 10 images!")
