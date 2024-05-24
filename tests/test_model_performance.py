import time

from source.model import ImageClassifier
from source.data_utils import construct_loader
from source.utils import fetch_image_paths_and_labels


def test_loaded_model_performance(checkpoint_file, data_dir):
    test_dir = data_dir

    paths, labels = fetch_image_paths_and_labels(test_dir)
    loader = construct_loader(paths, labels, batch_size=2, train=False)

    image_classifier = ImageClassifier()
    image_classifier.load_model(checkpoint_file)

    start_time = time.time()
    accuracy, loss, _ = image_classifier.evaluate_model(loader)
    end_time = time.time()

    evaluation_time = end_time - start_time

    assert loss < 2, "Loss is grater than 2."
    assert accuracy > 0.9, "Accuracy is less than 0.9"

    assert evaluation_time <= 2, f"Model evaluation took too long: {evaluation_time} seconds"


