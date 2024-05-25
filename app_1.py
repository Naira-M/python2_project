from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import base64
import torch
from torchvision import models, transforms
from PIL import Image
import io
from fastapi.staticfiles import StaticFiles
import torch.nn as nn
import os

app = FastAPI(
    title="Fake Face Image Detection API",
    description="Takes uploaded image and returns its class (fake or real).",
    version="0.0.3",
    openapi_tags=[{
        "name":
        "Classify",
        "description":
        "API endpoints related to image classification"
    }])

# Define the transform for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
### Here you can use the loading function which you have defined.
checkpoint = torch.load("checkpoint.pth", map_location=torch.device('cpu'))
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint)
model.eval()

# Directory to save the classified images and results
### Pay attension to this directory!!!
if not os.path.exists('classified_images'):
    os.makedirs('classified_images')


### Better and more attractive Greeting page!
@app.get("/", tags=["Greeting"], response_class=HTMLResponse)
async def root():
    """Greet a user."""
    return """
 <html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to Fake Face Image Detection API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-image: url('https://img.freepik.com/free-vector/realistic-polygonal-background_23-2148897123.jpg?size=626&ext=jpg&ga=GA1.1.2021396666.1716520252&semt=ais_user'); 
            background-size: cover;
            background-position: center;
            text-align: center;
            padding-top: 50px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background-color: rgba(255, 255, 255, 0.8); /* Add some transparency to make the text more readable */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333333;
        }
        p {
            color: #666666;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
        .upload-container {
            margin-top: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .upload-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
        }
        .upload-btn:hover {
            background-color: #45a049;
        }
        #imageInput {
            display: none;
        }
        #result {
            margin-top: 20px;
            color: #333333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to Fake Face Image Detection API</h1>
        <h4><p>This API allows you to classify uploaded images as real or generated (fake) using ResNet18 model.</p></h4>
    </div>

    <div class="upload-container">
        <input type="file" id="imageInput" name="image">
        <label for="imageInput" class="upload-btn">Select</label>
    </div>

    <div id="result"></div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const resultDiv = document.getElementById('result');

        imageInput.addEventListener('change', async () => {
            const formData = new FormData();
            formData.append('image', imageInput.files[0]);

            try {
                const response = await fetch('/classify/', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                resultDiv.innerText = `Filename: ${data.filename}, Class: ${data.class}`;
            } catch (error) {
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>
"""


### You can keep your version of this endpoint, with predefinied function for prediction.


@app.post("/classify/", tags=["Classify"])
async def classify(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail="File provided is not an image.")

    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")

        # Save the original image
        image_path = f"classified_images/{image.filename}"
        img.save(image_path)

        # Transform the image
        img_t = transform(img).unsqueeze(0)

        # Predict the label
        with torch.no_grad():
            output = model(img_t)
            _, predicted = torch.max(output, 1)
            img_class = "Fake" if predicted.item() == 1 else "Real"

        return {"filename": image.filename, "class": img_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


### Here I prefer "/model-info" instead of "/metadata", since it contains more information about the model.
@app.get("/model-info", tags=["Model"])
async def model_info():
    return {
        "algorithm_name": "ResNet18",
        "related_research_papers": ["https://arxiv.org/abs/1512.03385"],
        "version_number": "1.0",
        "training_date": "2022-05-01",
        "dataset_used": "140k-real-and-fake-faces",
        "accuracy": "96.6%",
        "loss": "0.034"
    }


### Retrieve the history of classified images.
@app.get("/history", tags=["History"])
async def image_history():
    files = os.listdir('classified_images')
    history = [file for file in files]
    return {"history": history}


### Retrieve a classified image.
@app.get("/img/{image_name}", tags=["Classified Images"])
async def get_classified_image(image_name: str):
    file_path = f"classified_images/{image_name}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")


### Upload and classify multiple images at once!
@app.post("/multiple/", tags=["Classify1"])
async def upload_multiple(images: list[UploadFile] = File(...)):
    results = []
    for image in images:
        if not image.content_type.startswith("image/"):
            results.append({
                "filename": image.filename,
                "error": "File provided is not an image."
            })
            continue

        try:
            img = Image.open(io.BytesIO(await image.read())).convert("RGB")
            img_t = transform(img).unsqueeze(0)
            image_path = f"classified_images/{image.filename}"
            img.save(image_path)

            with torch.no_grad():
                output = model(img_t)
                _, predicted = torch.max(output, 1)
                img_class = "Fake" if predicted.item() == 1 else "Real"

            results.append({"filename": image.filename, "class": img_class})
        except Exception as e:
            results.append({"filename": image.filename, "error": str(e)})

    return {"results": results}


# Mount a directory containing static files (like HTML) as a static directory
app.mount("/static", StaticFiles(directory="."), name="static")
