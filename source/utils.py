import random
import os
from glob import glob
import io

import matplotlib.pyplot as plt
from PIL import Image
import cv2

from source.data_utils import preprocess_data


def display_images(image_paths, labels):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        # Read the image
        img = cv2.imread(img_path)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Display the image
        axes[i].imshow(img)
        axes[i].axis('off')
        # Set the label based on the directory name
        label_str = 'Fake' if label == 1 else 'Real'
        axes[i].set_title(label_str)
    plt.tight_layout()


def fetch_image_paths_and_labels(image_dir):
    real_dir = os.path.join(image_dir, "real")
    fake_dir = os.path.join(image_dir, "fake")

    if not os.path.exists(real_dir) and not os.path.exists(fake_dir):
        raise FileNotFoundError(
            f"One or both required directories '{real_dir}' and '{fake_dir}' do not exist.")

    real_paths = sorted(glob(os.path.join(real_dir, "*.jpg")))
    fake_paths = sorted(glob(os.path.join(fake_dir, "*.jpg")))

    # Check if directories contain .jpg files
    if not real_paths and not fake_paths:
        raise ValueError(
            f"The directories '{real_dir}' and '{fake_dir}' are empty or does not contain any .jpg files.")

    real_labels = [0] * len(real_paths)
    fake_labels = [1] * len(fake_paths)
    data = list(zip(real_paths, real_labels)) + list(zip(fake_paths, fake_labels))
    random.shuffle(data)
    paths, labels = zip(*data)
    return paths, labels


def load_image(image_file):
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    transform = preprocess_data(False)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    return image
