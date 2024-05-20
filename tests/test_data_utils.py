import pytest
import numpy as np
import torch
from PIL import Image

from source.data_utils import preprocess_data


@pytest.mark.parametrize("train", [True, False])
def test_preprocess_data(train):

    sample_img_array = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
    sample_img_pil = Image.fromarray(np.transpose(sample_img_array, (1, 2, 0)))

    # Apply transformations on the sample dataset
    # For train and test mode be don't check the same things,
    # because Rotation and Flip are randomized
    transform = preprocess_data(train=train)
    transformed_img = transform(sample_img_pil)

    # Verify if transformations have been applied correctly
    # for transformed_img, original_img in zip(transformed_dataset, sample_dataset):
    assert isinstance(transformed_img, torch.Tensor), "Transformed image should be a tensor"
    assert transformed_img.shape[0] == 3, "Transformed image should have 3 channels (RGB)"
    assert transformed_img.shape[1:] == (224, 224), "Transformed image should have shape (224, 224)"

    # TODO: Add normalization check
