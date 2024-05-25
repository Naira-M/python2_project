# Fake Face Image Detection API

## Overview
The Fake Face Image Detection API is a FastAPI-based application designed to classify images as either fake or real. It uses Resnet18 model to analyze uploaded images and determine their authenticity.

## Running the Server
Start the server:
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`
python app/fake_face_detection_api.py
```
The server runs on `http://0.0.0.0:8000` with live reloading enabled.

## Additional Notes
- Ensure the model file `files/checkpoint.pth` and metadata file `files/metadata.pkl` are accessible.
- The API processes one image at a time.

## Running tests

```bash
pytest --image-paths="files/images/img1.jpg,files/images/img2.jpg" --checkpoint-path="files/checkpoint.pth" --data-dir="data/sample_data"
```
The tests can be run with the following command-line arguments (the given values are the defaults). If the tests are run without these arguments and the default paths do not exist on your system, the tests will be skipped




