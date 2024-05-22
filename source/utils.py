import random
import os
from glob import glob
import io

import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet18


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
    real_dir = os.path.join(image_dir, category, "real")
    fake_dir = os.path.join(image_dir, category, "fake")

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


def classify_image(model, image_file, device=torch.device('cpu')):
    image = load_image(image_file)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        value, predicted = torch.max(outputs, 1)
    img_class = "Fake" if predicted.item() == 1 else "Real"

    return img_class


def load_model(checkpoint_path, device=torch.device('cpu')):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    return model
