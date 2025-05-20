import os
import numpy as np
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import BoundaryAttack
import matplotlib.pyplot as plt
from PIL import Image

# Build & Load the Model 
def build_kitti_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(256, 512, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(8, activation='sigmoid')
    ])
    return model

model = build_kitti_model()
model.load_weights("/content/drive/MyDrive/gp2/kitti_multilabel_cnn.h5")
print("Model loaded!")

classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=8,
    input_shape=(256, 512, 3),
    clip_values=(0, 1),
    channels_first=False
)
print("Model wrapped with ART!")

#  Load Class Names 
class_names = np.load("/content/drive/MyDrive/gp2/class_names.npy", allow_pickle=True)

# Output Directories
os.makedirs("/content/drive/MyDrive/gp2/boundary_success_annotated", exist_ok=True)
os.makedirs("/content/drive/MyDrive/gp2/boundary_attacked_npy", exist_ok=True)


root_path = "/content/drive/MyDrive/gp2/processed_images"
all_files = sorted([f for f in os.listdir(root_path) if f.endswith(".npy")])[4000:]

used_pairs = set()
success_count = 0

for i in range(len(all_files)):
    sample1 = np.load(os.path.join(root_path, all_files[i]), allow_pickle=True).item()
    labels1 = [ann['class'] for ann in sample1['annotations']]
    if len(labels1) != 1:
        continue

    label_to_avoid = labels1[0]
    img1 = np.expand_dims(sample1["image"].astype(np.float32), axis=0)

    for j in range(i+1, len(all_files)):
        if (i, j) in used_pairs or (j, i) in used_pairs:
            continue

        sample2 = np.load(os.path.join(root_path, all_files[j]), allow_pickle=True).item()
        labels2 = [ann['class'] for ann in sample2['annotations']]
        if label_to_avoid not in labels2:
            img2 = np.expand_dims(sample2["image"].astype(np.float32), axis=0)

            pred_before = classifier.predict(img1)[0]
            top_idx_before = np.argmax(pred_before)
            label_before = class_names[top_idx_before]
            conf_before = pred_before[top_idx_before]

            attack = BoundaryAttack(
                estimator=classifier,
                targeted=False,
                max_iter=20,
                delta=0.01,
                epsilon=0.01,
                verbose=True
            )

            try:
                x_adv = attack.generate(x=img1, x_adv_init=img2)
                pred_after = classifier.predict(x_adv)[0]
                top_idx_after = np.argmax(pred_after)
                label_after = class_names[top_idx_after]
                conf_after = pred_after[top_idx_after]

                if label_before != label_after:
                    used_pairs.add((i, j))
                    success_count += 1
                    drop = conf_before - conf_after

                    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                    axs[0].imshow(img1[0])
                    axs[0].set_title(f"Original\n{label_before} ({conf_before:.2f})")

                    axs[1].imshow(x_adv[0])
                    axs[1].set_title(f"Adversarial\n{label_after} ({conf_after:.2f})")

                    diff = (x_adv[0] - img1[0]).clip(0, 1)
                    axs[2].imshow(diff * 100)
                    axs[2].set_title(f"Perturbation (x100)\nDrop: {drop:.4f}")
                    plt.colorbar(axs[2].imshow(diff * 100), ax=axs[2])

                    for ax in axs:
                        ax.axis("off")

                    plt.tight_layout()
                    vis_path = f"/content/drive/MyDrive/gp2/boundary_success_annotated/adv_{success_count:03}.png"
                    plt.savefig(vis_path)
                    plt.close()

                    # Save .npy of adversarial image
                    np.save(f"/content/drive/MyDrive/gp2/boundary_attacked_npy/adv_{success_count:03}.npy", x_adv[0])
                    break

            except Exception as e:
                print(f" Failed on pair {i}-{j}: {e}")
                continue

print(f" Total successful boundary attacks saved: {success_count}")
