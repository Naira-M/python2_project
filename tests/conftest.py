import pytest
import os


def pytest_addoption(parser):
    parser.addoption(
        "--image-paths",
        action="store",
        default="files/images/img1.jpg,files/images/img2.jpg",
        help="Comma-separated list of image paths")

    parser.addoption(
        "--checkpoint-path",
        action="store",
        default="files/checkpoint.pth",
        help="The latest trained model's checkpoint file.")

    parser.addoption(
        "--data-dir",
        action="store",
        default="data/sample_data",
        help="Directory of images.")


@pytest.fixture
def checkpoint_file(request):
    checkpoint_path = request.config.getoption("--checkpoint-path")
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Default checkpoint path '{checkpoint_path}' does not exist.")
    return checkpoint_path


@pytest.fixture
def data_dir(request):
    test_data_dir = request.config.getoption("--data-dir")
    if not os.path.exists(test_data_dir):
        pytest.skip(f"Default checkpoint path '{test_data_dir}' does not exist.")
    return test_data_dir


@pytest.fixture
def sample_data(request):
    image_paths_str = request.config.getoption("--image-paths")
    image_paths = image_paths_str.split(',')
    if len(image_paths) == 1:
        labels = [0]
    else:
        labels = [0] * (len(image_paths) // 2) + [1] * (len(image_paths) // 2)
    return image_paths, labels
