import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

data_dir = r"C:\Users\ASUS\Desktop\gp\processed_images"
model_save_path = r"C:\Users\ASUS\Desktop\gp\kitti_multilabel_cnn.h5"

print("Loading data...")

# Load images and labels
images = []
labels = []

for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith('.npy'):
        continue
    data = np.load(os.path.join(data_dir, fname), allow_pickle=True).item()
    images.append(data['image'])

    # Get all object classes in that image (multi-label)
    objs = [ann['class'] for ann in data['annotations'] if ann['class'] != 'DontCare']
    labels.append(objs)

print(f"✅ Loaded {len(images)} image-label pairs.")

# Binarize labels using multi-label binarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)
class_names = mlb.classes_
print(f"📚 Classes: {list(class_names)}")

# Convert images to NumPy array
X = np.array(images, dtype=np.float32)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN Model
model = models.Sequential([
    layers.Input(shape=(256, 512, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(class_names), activation='sigmoid')  # Multi-label output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary loss for multi-label
              metrics=['accuracy'])

# Train
print("Training started...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save model and classes
model.save(model_save_path)
np.save(r"C:\Users\ASUS\Desktop\gp\class_names.npy", class_names)

print(f"Model saved to {model_save_path}")