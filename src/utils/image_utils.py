import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        label = subfolder
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img = cv2.imread(os.path.join(subfolder_path, filename))
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

def preprocess_images(images):
    resized_images = [cv2.resize(img, (224, 224)) for img in images]
    preprocessed_images = np.array(resized_images, dtype=np.float32) / 255.0
    return preprocessed_images

def split_data(images, labels, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val
