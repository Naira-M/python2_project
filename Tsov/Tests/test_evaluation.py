import torch
import torch.nn as nn
from torchvision import models
from transform_and_loader import transform_and_loader
import torch
from evaluate_saved_model import evaluate_saved_model


def test_evaluation():
  image_paths = [
      '/content/pics/fake/ZZFRNG8UL2.jpg',
      '/content/pics/fake/ZZK7DY74LZ.jpg',
      '/content/pics/real/69995.jpg',
      '/content/pics/real/69991.jpg',
      '/content/pics/fake/ZZNKRXBE9B.jpg',
      '/content/pics/fake/ZZXYG0W3UZ.jpg',
      '/content/pics/real/69998.jpg',
      '/content/pics/real/69999.jpg',
  ]
  labels = [1, 1, 0, 0, 1, 1, 0, 0]  # Adjust according to the number of images

  dataloader = transform_and_loader(image_paths,
                                    labels,
                                    batch_size=2,
                                    train=False)

  checkpoint_path = "checkpoint.pth"
  model = models.resnet18()
  num_classes = 2
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  criterion = nn.CrossEntropyLoss()

  # Call the evaluation function
  test_loss, test_accuracy = evaluate_saved_model(model, dataloader, criterion,
                                                  checkpoint_path)

  # Check if the calculated accuracy is between 0 and 1
  assert 0 <= test_accuracy <= 1, "Accuracy should be between 0 and 1"

  # Check if the calculated loss is higher than 0
  assert test_loss > 0, "Loss should be higher than 0"
