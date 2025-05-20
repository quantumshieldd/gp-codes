Project File Documentation 

This repository contains all source code files used in the development, training, and testing of the adversarial attack detection system using classical and quantum models. Below is an overview of each important file and its purpose:

- **CNN.py**  
  Trains a convolutional neural network (CNN) model on preprocessed KITTI images using multi-label classification.

- **bdd100k_attacks.py**  
  Applies multiple adversarial attacks (FGSM, BIM, PGD) on the BDD100K dataset and saves the generated adversarial samples.

- **boundaryAttack.py**  
  Runs the Boundary Attack, a black-box adversarial attack targeting single-label images.

- **colorspace.py**  
  Implements a color space–based attack by modifying RGB channels to simulate subtle adversarial changes.

- **quantum_model.py**  
  Defines and trains a hybrid CNN quantum  model using PennyLane and TensorFlow for robust attack detection. 

- **patchattack.py**  
  Generates adversarial patch-based attacks which simulate real-world stickers and occlusions on images.

- **hopskipjump.py**  
  Applies the HopSkipJump black-box attack for generating adversarial samples with minimal queries.

- **gussian.py**  
  Adds Gaussian noise to input images for robustness testing and noise-based evasion attack simulation.

- **rainattack.py**  
  Simulates a weather-based attack by adding rain overlays to images and evaluating model robustness.

- **salt_and_pepper.py**  
  Applies salt-and-pepper noise to simulate image degradation and measure classifier reliability.

- **preprocessing_images.py**  
  Handles preprocessing of KITTI/BDD100K images: resizing, normalization, and annotation merging into `.npy` format.

- **preprocessing_quantum.py**  
  Custom preprocessing for the quantum model: downscales and reshapes images for compatibility with QVC layers.

- **preprocess_bdd100k.py**  
  Script for batch formatting and resizing BDD100K image dataset before attack generation.

- **class_names.npy**  
  Numpy file storing class labels for use during model prediction and evaluation.

- **cnn-qvc-api.zip**  
  Backend API (FastAPI/Flask) code that serves the trained model for live predictions via HTTP requests.

- **FrontEnd.zip**  
  ReactJS-based frontend interface (Quantum Shield) for uploading images and visualizing attack detection results.

> The file `kitti_multilabel_cnn.h5` is excluded from GitHub due to size restrictions but is available via Google Drive upon request. 
