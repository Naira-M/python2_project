import torch
import torchvision
from custom_dataset import CustomDataset
from torchvision import transforms
from transform_and_loader import transform_and_loader


def test_data_loading():
    image_paths = [
        '/content/pics/fake/ZZNKRXBE9B.jpg', '/content/pics/real/69995.jpg'
    ]
    labels = [0, 1]

    loader = transform_and_loader(image_paths,
                                  labels,
                                  batch_size=2,
                                  train=False)

    # Test __len__() method
    assert len(loader) == 1, "Length of loader should be 2"

    # Test data loading using DataLoader
    for img, label in loader:
        # Test __getitem__() method
        assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
        assert isinstance(label,
                          torch.Tensor), "Labels should be a torch.Tensor"
        assert img.shape[0] == label.shape[
            0], "Batch size should be consistent"
        assert img.shape == (2, 3, 224,
                             224), "Image shape should be (3, 224, 224)"
        assert (label[0] == 0 or label[0] == 1) and (
            label[1] == 0 or label[1] == 1), "Labels should be 0 or 1"
