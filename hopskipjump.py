import os
import numpy as np
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import HopSkipJump
import matplotlib.pyplot as plt
from PIL import Image

#Setup Model
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
print(" Wrapped with ART!")

# Load Class Names 
class_names = np.load("/content/drive/MyDrive/gp2/class_names.npy", allow_pickle=True)

#Output Directory 
os.makedirs("/content/drive/MyDrive/gp2/hsja_success_annotated", exist_ok=True)

#Find the last successful attack 
existing_files = sorted([f for f in os.listdir("/content/drive/MyDrive/gp2/hsja_success_annotated") if f.endswith(".png")])
success_count = len(existing_files)

#Process Images
root_path = "/content/drive/MyDrive/gp2/processed_images"
all_files = sorted([f for f in os.listdir(root_path) if f.endswith(".npy")])[4200:]


for i in range(success_count, min(700, len(all_files))):  # Start from the last successful index
    sample1 = np.load(os.path.join(root_path, all_files[i]), allow_pickle=True).item()
    image1 = sample1["image"].astype(np.float32)
    anns1 = sample1["annotations"]
    labels1 = [ann["class"] for ann in anns1]

    if len(labels1) != 1:
        continue

    label_to_avoid = labels1[0]

    for j in range(i+1, len(all_files)):
        sample2 = np.load(os.path.join(root_path, all_files[j]), allow_pickle=True).item()
        labels2 = [ann["class"] for ann in sample2["annotations"]]

        if label_to_avoid not in labels2:
            img1 = np.expand_dims(image1, axis=0)
            img2 = np.expand_dims(sample2["image"].astype(np.float32), axis=0)

            pred_before = classifier.predict(img1)[0]
            top_idx_before = np.argmax(pred_before)
            label_before = class_names[top_idx_before]
            conf_before = pred_before[top_idx_before]

            attack = HopSkipJump(
                classifier=classifier,
                targeted=False,
                max_iter=20,
                max_eval=1000,
                init_eval=100,
                init_size=10,
                verbose=False
            )

            try:
                x_adv = attack.generate(x=img1, x_adv_init=img2)
                pred_after = classifier.predict(x_adv)[0]
                top_idx_after = np.argmax(pred_after)
                label_after = class_names[top_idx_after]
                conf_after = pred_after[top_idx_after]

                if label_before != label_after:
                    success_count += 1
                    drop = conf_before - conf_after

                    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

                    axs[0].imshow(img1[0])
                    axs[0].set_title(f"Original\nLabel: {label_before}\nConfidence: {conf_before:.2f}")

                    axs[1].imshow(x_adv[0])
                    axs[1].set_title(f"Adversarial\nLabel: {label_after}\nConfidence: {conf_after:.2f}")

                    diff = (x_adv[0] - img1[0]).clip(0, 1)
                    axs[2].imshow(diff * 100)
                    axs[2].set_title(f"Perturbation (x100)\nDrop: {drop:.4f}")
                    plt.colorbar(axs[2].imshow(diff * 100), ax=axs[2])

                    for ax in axs:
                        ax.axis("off")

                    plt.tight_layout()
                    out_path = f"/content/drive/MyDrive/gp2/hsja_success_annotated/adv_{success_count:03}.png"
                    plt.savefig(out_path)
                    plt.close()

                    # Save the attacked image as well
                    attacked_image_path = f"/content/drive/MyDrive/gp2/hsja_success_annotated/attacked_{success_count:03}.png"
                    Image.fromarray(np.uint8(x_adv[0] * 255)).save(attacked_image_path)

                    # Update success count in a text file
                    with open("/content/drive/MyDrive/gp2/success_count.txt", "w") as file:
                        file.write(str(success_count))

                    break

            except Exception as e:
                print(f" Attack failed on {all_files[i]} -> {all_files[j]}: {e}")
                continue

print(f" Total successful attacks saved: {success_count}")
