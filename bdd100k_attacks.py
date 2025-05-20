import os
import numpy as np
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier
from tensorflow.keras.models import load_model
from tqdm import tqdm

model_path = r"C:\Users\eslam\Desktop\kitti\kitti_multilabel_cnn.h5"
data_folder = r"C:\Users\eslam\Desktop\bdd100k_preprocessed"
save_root = r"C:\Users\eslam\Desktop\bdd100k_attacks"
os.makedirs(save_root, exist_ok=True)

#Load Model
model = load_model(model_path)

#ART Wrapper 
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=3,
    input_shape=(256, 512, 3),
    loss_object=tf.keras.losses.BinaryCrossentropy()
)

# List of Attacks
attacks = {
    "fgsm": FastGradientMethod(estimator=classifier, eps=0.05),
    "bim": BasicIterativeMethod(estimator=classifier, eps=0.05, eps_step=0.01, max_iter=10),
    "pgd": ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=40)
}


npy_files = [f for f in os.listdir(data_folder) if f.endswith(".npy")]

for name, attack in attacks.items():
    output_dir = os.path.join(save_root, name)
    os.makedirs(output_dir, exist_ok=True)
    print(f" Starting {name.upper()} attack on {len(npy_files)} images...")

    for file in tqdm(npy_files, desc=f"{name.upper()} Attack"):
        img_path = os.path.join(data_folder, file)
        try:
            data = np.load(img_path, allow_pickle=True).item()
            image = np.expand_dims(data['image'], axis=0)  # shape: (1, 256, 512, 3)
            adv_img = attack.generate(x=image)
            # Save .npy file
            np.save(os.path.join(output_dir, file), {'image': adv_img[0], 'original_filename': file})
        except Exception as e:
            print(f" Error processing {file}: {e}")