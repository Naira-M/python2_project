import random
import os
from glob import glob

import matplotlib.pyplot as plt
import cv2
from PIL import Image

from source.data_utils import preprocess_data


def display_images(image_paths, labels):
    """
    Display images along with their labels.

    Parameters:
    image_paths: List of paths to the images.
    labels: List of labels corresponding to the images.
    """
    num_images = len(image_paths)
    num_rows = (num_images + 2) // 3  # Calculate number of rows for subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))
    axes = axes.flatten()

    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        # Read the image
        img = cv2.imread(img_path)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title('Fake' if label == 1 else 'Real')

    # Hide remaining empty subplots
    for j in range(num_images, num_rows * 3):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def fetch_image_paths_and_labels(image_dir, category):
    real_paths = sorted(glob(os.path.join(image_dir, category, "real", "*.jpg")))
    fake_paths = sorted(glob(os.path.join(image_dir, category, "fake", "*.jpg")))
    real_labels = [0] * len(real_paths)
    fake_labels = [1] * len(fake_paths)
    data = list(zip(real_paths, real_labels)) + list(zip(fake_paths, fake_labels))
    random.shuffle(data)
    paths, labels = zip(*data)
    return paths, labels


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = preprocess_data(False)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

