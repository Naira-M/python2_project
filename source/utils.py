import random
import os
from glob import glob
import io

from PIL import Image

from source.data_utils import preprocess_data


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
