import pytest
import os
from PIL import Image
import numpy as np


def pytest_addoption(parser):
    parser.addoption(
        "--image-paths",
        action="store",
        default="files/images/img1.jpg,files/images/img2.jpg",
        help="Comma-separated list of image paths")


@pytest.fixture
def sample_data(request):
    image_paths_str = request.config.getoption("--image-paths")

    image_paths = image_paths_str.split(',')
    if len(image_paths) == 1:
        labels = [0]
    else:
        labels = [0] * (len(image_paths) // 2) + [1] * (len(image_paths) // 2)
    return image_paths, labels


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

    empty_category_dir = tmp_image_dir.mkdir("empty_category")
    empty_category_dir.mkdir("real")
    empty_category_dir.mkdir("fake")

    return str(tmp_image_dir)


@pytest.fixture
def generate_images(tmpdir):
    np.random.seed(42)

    tmp_image_dir = tmpdir.mkdir("images")

    # Generate image paths and labels
    image_paths = []
    labels = []

    for i in range(8):
        img_path = os.path.join(tmp_image_dir, f"image_{i}.jpg")
        random_array = np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8)
        image = Image.fromarray(random_array.transpose(1, 2, 0))
        image.save(img_path)
        image_paths.append(img_path)
        labels.append(np.random.randint(0, 2))

    return tmp_image_dir, image_paths, labels

