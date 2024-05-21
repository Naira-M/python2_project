from image_utils import fetch_image_paths_and_labels
import os


def test_fetch_image_paths_and_labels():
    image_dir = 'pics'
    paths, labels = fetch_image_paths_and_labels(image_dir)

    assert len(paths) > 0, "No image paths found"
    assert all(label in [0, 1]
               for label in labels), "All labels should be 0 or 1"
    assert len(paths) == len(
        labels), "Number of paths and labels should be equal"


# 2. This test verifies that all real images are labeled as 0 and all fake images are labeled as 1.
def test_labeling_accuracy():
    image_dir = 'pics'
    paths, labels = fetch_image_paths_and_labels(image_dir)

    for path, label in zip(paths, labels):
        if 'real' in path:
            assert label == 0, f"Expected label 0 for real image, got {label}"
        elif 'fake' in path:
            assert label == 1, f"Expected label 1 for fake image, got {label}"


# 3. This test checks if the function correctly filters out images with unsupported extensions (only JPEG images are allowed in this case):
def test_image_extensions():
    image_dir = 'pics'
    paths, _ = fetch_image_paths_and_labels(image_dir)

    for path in paths:
        ext = os.path.splitext(path)[1]
        assert ext.lower() == '.jpg', f"Expected JPEG image, got {ext}"


# 4. This test compares the output of the function for two different runs and ensures that the paths are shuffled randomly.
def test_random_shuffling():
    image_dir = 'pics'
    paths1, _ = fetch_image_paths_and_labels(image_dir)
    paths2, _ = fetch_image_paths_and_labels(image_dir)

    assert paths1 != paths2, "Paths are not shuffled"
