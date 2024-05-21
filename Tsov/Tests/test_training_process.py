import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def test_training_with_generated_data():
    # Create random data
    torch.manual_seed(42)
    generated_images = torch.randn(8, 3, 224,
                                   224)  # 8 images, 3 channels, 224x224
    generated_labels = torch.randint(0, 2, (8, ))  # 8 labels, either 0 or 1

    # Define a simple model
    model = models.resnet18(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(generated_images)
    loss = criterion(outputs, generated_labels)
    loss.backward()
    optimizer.step()

    # Check if the output is valid
    assert outputs.shape == (8, 2), "Output shape should be (8, 2)"
    assert loss.item() > 0, "Loss should be greater than 0"
    assert loss.item() < 10, "Loss should be at least less than 10"
