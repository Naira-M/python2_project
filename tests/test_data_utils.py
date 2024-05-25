import os
import pytest
import numpy as np
import torch
from PIL import Image

from source.data_utils import preprocess_data, construct_loader


@pytest.mark.parametrize("train", [True, False])
def test_preprocess_data(train):

    sample_img_array = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
    sample_img_pil = Image.fromarray(np.transpose(sample_img_array, (1, 2, 0)))

    # Apply transformations on the sample dataset
    # For train and test modes check the same things,
    # because Rotation and Flip are randomized
    transform = preprocess_data(train=train)
    transformed_img = transform(sample_img_pil)

    # Verify if transformations have been applied correctly
    assert isinstance(transformed_img, torch.Tensor), "Transformed image should be a tensor"
    assert transformed_img.shape[0] == 3, "Transformed image should have 3 channels (RGB)"
    assert transformed_img.shape[1:] == (224, 224), "Transformed image should have shape (224, 224)"


@pytest.mark.parametrize("batch_size", [1, 2, 33])
def test_data_loading(sample_data, batch_size):
    image_paths, labels = sample_data

    # Check if all image files exist
    all_files_exist = all(os.path.isfile(path) for path in image_paths)
    if not all_files_exist:
        pytest.skip("Some image files do not exist. Please provide valid file paths.")

    loader = construct_loader(image_paths, labels, batch_size=batch_size, train=False)

    for img, label in loader:
        assert img.shape[0] == label.shape[0], "Batch size should be consistent"
        assert all(l in [0, 1] for l in label), "Labels should be 0 or 1"

