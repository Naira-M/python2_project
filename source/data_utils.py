import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset


def preprocess_data(train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


def construct_loader(image_paths, labels, batch_size=32, train=True):
    """
    Function to apply transformations and convert to DataLoader format
    """
    transform = preprocess_data(train)
    dataset = CustomDataset(image_paths, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader

