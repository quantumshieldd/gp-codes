import os
import numpy as np
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod
from art.estimators.classification import TensorFlowV2Classifier
from tensorflow.keras.models import load_model
from tqdm import tqdm 


processed_images_dir = r"C:\Users\eslam\Desktop\kitti\processed_images"
model_path = r"C:\Users\eslam\Desktop\kitti\kitti_multilabel_cnn.h5"
class_names_path = r"C:\Users\eslam\Desktop\kitti\class_names.npy"
output_root_dir = r"C:\Users\eslam\Desktop\kitti\processed_images_adversarial"

#Load model and class names 
print("🔁 Loading model...")
model = load_model(model_path)
class_names = np.load(class_names_path, allow_pickle=True)

#Wrap with ART Classifier 
print(" Wrapping with ART TensorFlowV2Classifier...")
classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=tf.keras.losses.BinaryCrossentropy(),
    nb_classes=len(class_names),
    input_shape=(256, 512, 3),
    clip_values=(0.0, 1.0)
)

#Define attacks 
attacks = {
    'FGSM': FastGradientMethod(estimator=classifier, eps=0.03),
    'PGD': ProjectedGradientDescent(estimator=classifier, eps=0.03, max_iter=40),
    'BIM': BasicIterativeMethod(estimator=classifier, eps=0.03, max_iter=10)
}

filenames = sorted([f for f in os.listdir(processed_images_dir) if f.endswith('.npy')])
batch_size = 32

print(f" Found {len(filenames)} files. Starting batch attack generation...")

#Process in batches
for i in tqdm(range(0, len(filenames), batch_size), desc="🔄 Processing Batches"):
    batch_files = filenames[i:i + batch_size]
    samples = []

    # Load batch data
    for fname in batch_files:
        data_path = os.path.join(processed_images_dir, fname)
        data = np.load(data_path, allow_pickle=True).item()
        samples.append((fname, data['image'], data['annotations']))

    x_batch = np.array([img for _, img, _ in samples])

    for attack_name, attack in attacks.items():
        print(f"⚔ Running {attack_name} on batch {i // batch_size + 1}...")
        x_adv = attack.generate(x=x_batch)

        # Save outputs
        attack_dir = os.path.join(output_root_dir, attack_name)
        os.makedirs(attack_dir, exist_ok=True)

        for j, (fname, _, annotations) in enumerate(samples):
            adv_sample = {
                'image': x_adv[j],
                'annotations': annotations,
                'attack_type': attack_name
            }

            out_path = os.path.join(attack_dir, fname)
            np.save(out_path, adv_sample)

print("All adversarial examples successfully generated and saved!")
