import argparse

from source.model import ImageClassifier
from source.utils import fetch_image_paths_and_labels
from source.data_utils import construct_loader


def main(data_path, checkpoint_path):
    # Load test data
    test_image_paths, test_labels = fetch_image_paths_and_labels(data_path)

    # Load trained model
    print("Loading model...")
    image_classifier = ImageClassifier()
    image_classifier.load_model(checkpoint_path)

    print("Making DataLoader...")
    test_loader = construct_loader(test_image_paths, test_labels)

    # Evaluate the model
    print("Starting model evaluation...")
    test_accuracy, test_loss, _ = image_classifier.evaluate_model(test_loader)

    print("Test Accuracy:", test_accuracy)
    print("Test Loss:", test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data.")
    parser.add_argument("data_path", type=str,
                        help="Path to the directory containing 'test' directory for data․")
    parser.add_argument("checkpoint_path", type=str,
                        help="Path to the checkpoint file for the trained model․")
    args = parser.parse_args()

    main(args.data_path, args.checkpoint_path)
