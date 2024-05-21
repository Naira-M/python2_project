import os
from glob import glob
import random

def fetch_image_paths_and_labels(image_dir):
    real_paths = sorted(glob(os.path.join(image_dir, "real", "*.jpg")))
    fake_paths = sorted(glob(os.path.join(image_dir, "fake", "*.jpg")))
    real_labels = [0] * len(real_paths)
    fake_labels = [1] * len(fake_paths)
    data = list(zip(real_paths, real_labels)) + list(zip(fake_paths, fake_labels))
    random.shuffle(data)
    paths, labels = zip(*data)
    return paths, labels