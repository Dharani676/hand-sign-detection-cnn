import os
import cv2
import numpy as np

IMG_SIZE = 64

def load_data(data_dir):
    data = []
    labels = []

    label_names = sorted([l for l in os.listdir(data_dir) if not l.startswith('.')])
    label_map = {label: i for i, label in enumerate(label_names)}
    for label in label_names:
        path = os.path.join(data_dir, label)

        for img in os.listdir(path):
            if img.startswith('.'):
                continue

            img_path = os.path.join(path, img)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 🔥 IMPORTANT FIX
            if image is None:
                print("Failed to load:", img_path)
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            data.append(image)
            labels.append(label_map[label])
 
    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    labels = np.array(labels)

    return data, labels, label_map