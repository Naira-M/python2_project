import pytest
import os

from source.model import ImageClassifier

from source.data_utils import construct_loader


def test_train_model(generate_images):
    tmp_image_dir, image_paths, labels = generate_images

    loader = construct_loader(image_paths, labels, batch_size=4, train=True)

    # Temporary files for checkpoint and metadata
    checkpoints = os.path.join(tmp_image_dir, "checkpoint.pth")
    metadata = os.path.join(tmp_image_dir, "metadata.pkl")

    image_classifier = ImageClassifier()
    loss = image_classifier.train_model(loader, loader, checkpoints, metadata, num_epochs=1, device='cpu')

    # Assert that the loss is a scalar value
    assert isinstance(loss, float), "Loss should have type float."
    assert loss > 0, "Loss should be greater than 0."

    # Check output shape
    for images, _ in loader:
        outputs = image_classifier.model(images)
        # batch size is 4 and the number of classes is 2
        assert outputs.shape == (4, 2), "Output shape should be (batch size, num_classes)"


def test_evaluate_model(generate_images):

    _, image_paths, labels = generate_images

    loader = construct_loader(image_paths, labels, batch_size=4, train=False)

    image_classifier = ImageClassifier()
    test_accuracy, test_loss, misclassified_images = image_classifier.evaluate_model(loader)

    # Check if the calculated accuracy is between 0 and 1
    assert 0 <= test_accuracy <= 1, "Accuracy should be between 0 and 1"

    # Check if the calculated loss is higher than 0
    assert test_loss > 0, "Loss should be higher than 0"


def test_predict(generate_images):
    _, image_paths, _ = generate_images

    image_classifier = ImageClassifier()
    with open(image_paths[0], "rb") as img:
        img_class = image_classifier.predict(img)

    assert isinstance(img_class, str), "Class name should be a string."
    assert img_class == "Fake" or img_class == "Real", "Class is not 'Fake' or 'Real'"


def test_load_model_invalid_checkpoint():
    checkpoint_path = "nonexistent_checkpoint.pth"

    image_classifier = ImageClassifier()

    with pytest.raises(FileNotFoundError):
        image_classifier.load_model(checkpoint_path)


