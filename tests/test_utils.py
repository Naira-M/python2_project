import pytest
import os
import torch
from PIL import Image, UnidentifiedImageError

from source.utils import fetch_image_paths_and_labels, load_image


@pytest.fixture
def temp_image_dir(tmpdir):
    # Create a temporary directory
    tmp_image_dir = tmpdir.mkdir("images")

    # Create 'real' and 'fake' subdirectories for each category
    for category in ["train", "test"]:
        category_dir = tmp_image_dir.mkdir(category)
        real_dir = category_dir.mkdir("real")
        fake_dir = category_dir.mkdir("fake")

        # Create sample image files in 'real' and 'fake' subdirectories
        for i in range(5):
            open(os.path.join(real_dir, f"real_{i}.jpg"), 'a').close()
            open(os.path.join(fake_dir, f"fake_{i}.jpg"), 'a').close()

    # Create not empty image files
    category_dir = tmp_image_dir.mkdir("val")
    real_dir = category_dir.mkdir("real")
    real_img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    real_img.save(os.path.join(real_dir, "real_0.jpg"))

    return str(tmp_image_dir)


@pytest.mark.parametrize("category", ["train", "test"])
def test_fetch_image_paths_and_labels(temp_image_dir, category):
    image_dir = temp_image_dir
    paths, labels = fetch_image_paths_and_labels(image_dir, category)

    assert len(paths) == len(labels), "The number of paths should be equal to the number of labels."
    assert len(paths) == 10, "The number of paths and labels should be 10 (5 real + 5 fake)."
    assert all(isinstance(label, int) for label in labels), "All labels should be integers."
    assert set(labels) == {0, 1}, "Labels should contain both 0 (real) and 1 (fake)."
    assert len(paths) == len(set(paths)), "The paths should be different from each other."


def test_load_image(temp_image_dir):
    # Test loading of a single image using load_image function
    image_path = os.path.join(temp_image_dir, "val", "real", "real_0.jpg")
    image = load_image(image_path)

    assert image.shape == (1, 3, 224, 224), "Loaded image should have shape (1, 3, 224, 224)"
    assert image.dtype == torch.float32, "Loaded image should be of type float32"


def test_load_image_invalid_file(temp_image_dir):
    # Test loading of an empty image file
    invalid_image_path = os.path.join(temp_image_dir, "train", "real", "real_0.jpg")
    with pytest.raises(UnidentifiedImageError):
        load_image(invalid_image_path)
