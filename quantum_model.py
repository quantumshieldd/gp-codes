"""
Fine-tuning script for CNN + QVC model on KITTI adversarial detection dataset.
- Loads the original CNN + QVC model that achieved 88% accuracy (weights: maryam12.h5).
- Fine-tunes with new preprocessed data to reach 90%+ validation accuracy.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import pennylane as qml
from pennylane.qnn import KerasLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')

# Load new preprocessed data
X = np.load("G:/My Drive/X1_preprocessed.npy")
y = np.load("G:/My Drive/y1_labels.npy")
print(f" Loaded data shapes:\nImages: {X.shape}\nLabels: {y.shape}")
print(f"Pixel value range: Min={X.min():.2f}, Max={X.max():.2f}")
print("Dataset class distribution:", np.bincount(y))

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train = X_train / 255.0
X_test = X_test / 255.0
print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))

# Data Augmentation (same as original)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
train_generator = train_datagen.flow(X_train, y_train, batch_size=16)

# Quantum Circuit (same as original)
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Create the quantum layer
weight_shapes = {"weights": (3, n_qubits, 3)}
q_layer = KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

# Function to build the model architecture (used for both training and fallback loading)
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(256, 512, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_qubits, activation="tanh"),
        q_layer,
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# Build the model for training
model = build_model()

# Load the original weights
model.load_weights(r"C:\Users\ASUS\maryam12.h5")  

# Make all layers trainable for fine-tuning
model.trainable = True

# Compile with a small learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
checkpoint_full = callbacks.ModelCheckpoint(
    "maryam12_finetuned.h5",  # Save full model
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
checkpoint_weights = callbacks.ModelCheckpoint(
    "maryam12_finetuned_weights.h5",  # Save weights separately
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Fine-tune
epochs = 50
steps_per_epoch = len(X_train) // 16
history = model.fit(
    train_generator,
    validation_data=(X_test, y_test),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[early_stop, checkpoint_full, checkpoint_weights],
    verbose=1
)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Redefine q_layer for fallback loading
q_layer = KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)

# Attempt to load the full model, with fallback to weights
try:
    # Try loading the full model
    model = keras.models.load_model("testingquantum.h5", custom_objects={'KerasLayer': KerasLayer})
    print(" Successfully loaded the full model")
except Exception as e:
    print(f"Failed to load full model: {e}")
    print("Falling back to loading weights...")
    # Rebuild the model architecture
    model = build_model()
    # Load the weights
    model.load_weights("testingquantum_weights.h5")
    print("Successfully loaded weights into rebuilt model")

# Compile the model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])

# Evaluate
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nBest Training Accuracy: {train_acc:.4f}")
print(f" Best Validation Accuracy: {test_acc:.4f}")

# Classification Report
attack_types = ['Clean', 'Adversarial']
preds = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_test, preds, target_names=attack_types))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Inspect misclassified images
misclassified_idx = np.where(preds.flatten() != y_test.flatten())[0]
plt.figure(figsize=(10, 5))
for i, idx in enumerate(misclassified_idx[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx])
    plt.title(f"Pred: {preds[idx][0]}, True: {y_test[idx][0]}")
    plt.axis('off')
plt.show()

# Training Plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()