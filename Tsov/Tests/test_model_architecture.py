import torch
import torchvision
import torchvision.models as models


def test_model_architecture():
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Number of output classes for binary classification
    num_classes = 2

    # Instantiate the ResNet18 model
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the device
    model.to(device)

    # Get the number of output features of the final fully connected layer
    num_output_features = model.fc.out_features

    # Assert that the number of output features matches the expected number of classes
    assert num_output_features == num_classes, f"Number of output features should be {num_classes} for binary classification"
