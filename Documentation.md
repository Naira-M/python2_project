# CustomDataset.py 

## Overview
`CustomDataset` is a PyTorch `Dataset` subclass for handling image data, facilitating image loading and transformations.

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


# data_utils.py 

## Overview
Utility functions for preprocessing image data and constructing data loaders using the `CustomDataset` class.

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


# model.py 

## Overview
Defines the `ImageClassifier` class for image classification using a modified ResNet18 architecture.

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


# utils.py 

## Overview
Utility functions for handling images, fetching data, and visualization.

## Functions

### `display_images`
- **Parameters**: `image_paths`, `labels`.
- **Description**: Displays a set of images with labels.

### `fetch_image_paths_and_labels`
- **Parameters**: `image_dir`, `category`.
- **Returns**: `paths`, `labels`.
data/sample_data
### `load_image`
- **Parameters**: `image_file`.
- **Returns**: `image`.
