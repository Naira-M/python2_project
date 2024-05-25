import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import random
from glob import glob
from PIL import Image



def display_images(image_paths, labels):
  fig, axes = plt.subplots(3, 3, figsize=(10, 10))
  axes = axes.flatten()
  for i, (img_path, label) in enumerate(zip(image_paths, labels)):
      # Read the image
      img = cv2.imread(img_path)
      # Convert BGR to RGB
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # Display the image
      axes[i].imshow(img)
      axes[i].axis('off')
      # Set the label based on the directory name
      label_str = 'Fake' if label == 1 else 'Real'
      axes[i].set_title(label_str)
  plt.tight_layout()


def fetch_image_paths_and_labels(image_dir):
    real_paths = sorted(glob(os.path.join(image_dir, "real", "*.jpg")))
    fake_paths = sorted(glob(os.path.join(image_dir, "fake", "*.jpg")))
    real_labels = [0] * len(real_paths)
    fake_labels = [1] * len(fake_paths)
    data = list(zip(real_paths, real_labels)) + list(zip(fake_paths, fake_labels))
    random.shuffle(data)
    paths, labels = zip(*data)
    return paths, labels


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# Function to apply transformations
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

 # Function to convert to DataLoader format
def construct_loader(image_paths, labels, batch_size=32, train=True):
    transform=preprocess_data(train)
    dataset = CustomDataset(image_paths, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader



from torchvision.utils import make_grid

def display_normalized_images(train_loader, n=9):
    # Fetch the first batch of images
    images, labels = next(iter(train_loader))

    images = images[:n]
    labels = labels[:n]

    # Create a grid of images with 3 rows and 3 columns
    grid_img = make_grid(images, nrow=3)

    # Convert the tensor grid image to a numpy array and transpose it to (height, width, channels)
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # Clip pixel values to ensure they fall within the valid range
    grid_img = np.clip(grid_img, 0, 1)

    # Display the grid of images
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img)
    plt.axis('off')
