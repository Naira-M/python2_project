# Fake Face Image Detection API

## Overview
The Fake Face Image Detection API is a FastAPI-based application designed to classify images as either fake or real. It uses a pre-trained model to analyze uploaded images and determine their authenticity.

## Requirements
- **FastAPI**
- **Uvicorn**: ASGI server for running the application
- **Pickle**: For loading model metadata
- **ImageClassifier**: Custom model for image classification

## Installation
Install dependencies with:
```bash
pip install fastapi uvicorn pickle
```

## Running the Server
Start the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
The server runs on `http://0.0.0.0:8000` with live reloading enabled.

## API Endpoints

### Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "message": "Welcome to the Fake Face Image Detection API"
  }
  ```

### Model Metadata
- **URL**: `/metadata`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "model_name": "ImageClassifier",
    "accuracy": 0.95,
    "training_date": "2023-01-01"
  }
  ```

### Image Classification
- **URL**: `/classify/`
- **Method**: `POST`
- **Parameters**: `image` (image/jpeg, image/png)
- **Response**:
  ```json
  {
    "filename": "uploaded_image.jpg",
    "classification": "real"
  }
  ```

## Error Handling
- `200 OK`: Request successful.
- `400 Bad Request`: Invalid image format.
- `500 Internal Server Error`: Error processing the image.

## Additional Notes
- Ensure the model file `files/checkpoint.pth` and metadata file `files/metadata.pkl` are accessible.
- The API processes one image at a time.

## Example Requests

### cURL Examples
#### Root Endpoint
```bash
curl -X GET "http://0.0.0.0:8000/"
```

#### Model Metadata
```bash
curl -X GET "http://0.0.0.0:8000/metadata"
```

#### Image Classification
```bash
curl -X POST "http://0.0.0.0:8000/classify/" -F "image=@path_to_image.jpg"
```

---

# CustomDataset.py Documentation

## Overview
`CustomDataset` is a PyTorch `Dataset` subclass for handling image data, facilitating image loading and transformations.

## Requirements
- **Pillow**
- **PyTorch**

## Installation
```bash
pip install Pillow torch
```

## Class Documentation

### `CustomDataset`

#### Initialization
- **Parameters**:
  - `image_paths` (list of str): File paths to images.
  - `labels` (list): Corresponding labels.
  - `transform` (callable, optional): Transformations to apply to the images.

#### Methods

##### `__len__`
- **Returns**: `int` - Number of images.

##### `__getitem__`
- **Parameters**: `idx` (int) - Index of the data point.
- **Returns**: `tuple` - `(img, label)`.

## Example Usage
```python
from torchvision import transforms
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
labels = [0, 1]

dataset = CustomDataset(image_paths=image_paths, labels=labels, transform=transformations)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

for images, labels in data_loader:
    print(images.shape, labels)
```

---

# data_utils.py Documentation

## Overview
Utility functions for preprocessing image data and constructing data loaders using the `CustomDataset` class.

## Requirements
- **torchvision**
- **PyTorch**

## Installation
```bash
pip install torch torchvision
```

## Functions

### `preprocess_data`
- **Parameters**: `train` (bool) - Indicates if the transformations are for training.
- **Returns**: `transform` (callable).

### `construct_loader`
- **Parameters**:
  - `image_paths` (list of str): File paths to images.
  - `labels` (list): Corresponding labels.
  - `batch_size` (int, optional): Defaults to 32.
  - `train` (bool, optional).
- **Returns**: `DataLoader`.

## Example Usage
```python
from data_utils import preprocess_data, construct_loader

train_image_paths = ['path/to/train/image1.jpg', 'path/to/train/image2.jpg']
train_labels = [0, 1]
train_transforms = preprocess_data(train=True)
train_loader = construct_loader(train_image_paths, train_labels, batch_size=32, train=True)

for images, labels in train_loader:
    # training routine here
    pass
```

---

# model.py Documentation

## Overview
Defines the `ImageClassifier` class for image classification using a modified ResNet18 architecture.

## Requirements
- **Python 3.x**
- **PyTorch**
- **Torchvision**
- **Pillow**
- **Pickle**

## Installation
```bash
pip install torch torchvision Pillow
```

## Class Documentation

### `ImageClassifier`

#### Initialization
- **Parameters**: `num_classes` (int, optional) - Defaults to 2.

#### Methods

##### `train_model`
- **Parameters**: `train_loader`, `test_loader`, `checkpoint_path`, `metadata_path`, `num_epochs`, `device`.
- **Returns**: `best_valid_loss` (float).

##### `evaluate_model`
- **Parameters**: `test_loader`, `device`.
- **Returns**: `test_accuracy`, `test_loss`, `misclassified_images`.

##### `predict`
- **Parameters**: `image_file`, `device`.
- **Returns**: `img_class`.

##### `load_model`
- **Parameters**: `checkpoint_path`, `device`.

## Example Usage
```python
from model import ImageClassifier
from custom_dataset import CustomDataset
from data_utils import preprocess_data, construct_loader
from torch.utils.data import DataLoader

train_image_paths = ['path/to/train/image1.jpg', 'path/to/train/image2.jpg']
train_labels = [0, 1]
train_transforms = preprocess_data(train=True)
train_loader = DataLoader(CustomDataset(train_image_paths, train_labels, transform=train_transforms), batch_size=32, shuffle=True)

classifier = ImageClassifier(num_classes=2)
checkpoint_path = 'path/to/checkpoint.pth'
metadata_path = 'path/to/metadata.pkl'
best_loss = classifier.train_model(train_loader, test_loader, checkpoint_path, metadata_path)

classifier.load_model(checkpoint_path)
image_path = 'path/to/image.jpg'
classification_result = classifier.predict(image_path)
print(f'The image is classified as: {classification_result}')
```

---

# utils.py Documentation

## Overview
Utility functions for handling images, fetching data, and visualization.

## Requirements
- **Python 3.x**
- **OpenCV**
- **Matplotlib**
- **Pillow**

## Installation
```bash
pip install opencv-python-headless matplotlib Pillow
```

## Functions

### `display_images`
- **Parameters**: `image_paths`, `labels`.
- **Description**: Displays a set of images with labels.

### `fetch_image_paths_and_labels`
- **Parameters**: `image_dir`, `category`.
- **Returns**: `paths`, `labels`.

### `load_image`
- **Parameters**: `image_file`.
- **Returns**: `image`.

## Example Usage
```python
from utils import display_images, fetch_image_paths_and_labels, load_image

image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
labels = [0, 1]
display_images(image_paths, labels)
```

---

# conftest.py Documentation

## Overview
Configures pytest fixtures for testing the image classification application.

## Requirements
- **Python 3.x**
- **Pytest**
- **Pillow**
- **NumPy**

## Installation
```bash
pip install pytest Pillow numpy
```

## Fixtures

### `checkpoint_file`
- **Purpose**: Ensures the checkpoint path exists.

### `data_dir`
- **Purpose**: Verifies the data directory exists.

### `sample_data`
- **Purpose**: Splits image paths into labels and paths.

### `temp_image_dir`
- **Purpose**: Creates a temporary directory for testing.

### `generate_images`
- **Purpose**: Generates random images and assigns labels.

## Example Usage in Test
```python
import pytest
from model import ImageClassifier

@pytest.fixture
def checkpoint_file(pytestconfig):
    checkpoint_path = pytestconfig.getoption("checkpoint_path")
    if not os.path.exists(checkpoint_path):
        pytest.skip("Checkpoint path does not exist")
    return checkpoint_path

@pytest.fixture
def sample_data(pytestconfig):
    image_paths = pytestconfig.getoption("image_paths")
    labels = [0] * (len(image_paths) // 2) + [1] * (len(image_paths) // 2)
    return image_paths, labels

def test_image_classification(checkpoint_file, sample_data):
    classifier = ImageClassifier()
    classifier.load_model(checkpoint_file)
    
    image_paths, labels = sample_data
   
