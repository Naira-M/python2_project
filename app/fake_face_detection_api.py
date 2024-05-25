import pickle
import os
import io

from PIL import Image

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse


from source.model import ImageClassifier


MODEL = ImageClassifier()

# Create dir for saving images
# if not os.path.exists('classified_images'):
#     os.makedirs('classified_images')

app = FastAPI(
    title="Fake Face Image Detection API",
    description="Takes uploaded image and returns its class (fake or real).",
    version="0.0.1",
    openapi_tags=[{"name": "Classify", "description": "API endpoints related to image classification"}]
)


@app.get("/", tags=["Greeting"], response_class=HTMLResponse)
def root():
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
                background-image: url(
                'https://img.freepik.com/free-vector/realistic-polygonal-background_23-2148897123.jpg?size=626&ext=jpg&ga=GA1.1.2021396666.1716520252&semt=ais_user'); 
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


@app.get("/model-info", tags=["Model"])
def model_metadata():
    """
    Retrieve information about the latest trained model.

    This endpoint reads the metadata of the latest trained model from a pickle file
    and returns it as a JSON response.

    Returns:
        JSONResponse: A JSON response containing the model metadata.
    """
    with open("files/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return JSONResponse(content=metadata)


@app.post("/classify/", tags=["Model"])
def classify(
        image: UploadFile = File(description="A required image file for classification.")):
    """
    Classify an uploaded image as real or generated (fake).

    This endpoint accepts an image file, checks its type, and uses a pre-trained model
    to classify the image as real or generated. The classification result is returned
    in a JSON response.

    Args:
        image (UploadFile): The image file to be classified.

    Returns:
        JSONResponse: A JSON response containing the filename and classification result,
                      or an error message if the file is not an image.
    """
    if not image.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

    # Save the original image
    # img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    # image_path = f"classified_images/{image.filename}"
    # img.save(image_path)

    try:
        MODEL.load_model("files/checkpoint.pth")
        img_class = MODEL.predict(image.file)
        return JSONResponse(content={
            "filename": image.filename,
            "class": img_class
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify_multiple_images/", tags=["Model"])
def upload_multiple(
        images: list[UploadFile] = File(...)):
    """
    Classify multiple uploaded images as real or generated (fake).

    This endpoint accepts a list of image files, checks their types, and uses a pre-trained model
    to classify each image as real or generated. The classification results are returned in a JSON response.

    Args:
        images (list[UploadFile]): A list of image files to be classified.

    Returns:
        dict: A dictionary containing the filenames and classification results for each image,
              or error messages for files that are not images.
    """
    results = []
    for image in images:
        if not image.content_type.startswith("image/"):
            results.append({"filename": image.filename, "error": "File provided is not an image."})
            continue

        # Save the original image
        # img = Image.open(io.BytesIO(image.read())).convert("RGB")
        # image_path = f"classified_images/{image.filename}"
        # img.save(image_path)

        try:
            MODEL.load_model("files/checkpoint.pth")
            img_class = MODEL.predict(image.file)
            results.append({"filename": image.filename, "class": img_class})

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"results": results}


# @app.get("/history", tags=["History"])
# def image_history():
#     """
#
#     """
#     files = os.listdir("classified_images")
#     history = [file for file in files]
#     if history:
#         return {"history": history}
#     else:
#         return "There are no images in history."
#
#
# @app.get("/img/{image_name}", tags=["History"])
# def get_classified_image(image_name: str):
#     """
#
#     """
#     file_path = f"classified_images/{image_name}"
#     if os.path.exists(file_path):
#         return FileResponse(file_path)
#     else:
#         raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    uvicorn.run("app.fake_face_detection_api:app", host="0.0.0.0", port=8000, reload=True)

