import pickle
import os

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse


from source.model import ImageClassifier


MODEL = ImageClassifier()

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
   <html>
    <head>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to Fake Face Image Detection API</h1>
            <h4><p>This API allows you to classify uploaded images as real or generated (fake) using ResNet18 model.</p></h4>
            <p><a href="/classify/">Upload an image to get started!</a></p>
        </div>
    </body>
    </html>
    """


@app.get("/model-info", tags=["Model"])
def model_metadata():
    """
    Information of the latest trained mode.
    """
    with open("files/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return JSONResponse(content=metadata)


@app.post("/classify/", tags=["Model"])
def classify(
        image: UploadFile = File(description="A required image file for classification.")):
    """
    Finds out if the uploaded image is real or generated (fake).
    """
    if not image.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

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
async def upload_multiple(
        images: list[UploadFile] = File(...)):
    """
        Finds out if the uploaded images are real or generated (fake).
    """
    results = []
    for image in images:
        if not image.content_type.startswith("image/"):
            results.append({"filename": image.filename, "error": "File provided is not an image."})
            continue

        try:
            MODEL.load_model("files/checkpoint.pth")
            img_class = MODEL.predict(image.file)
            results.append({"filename": image.filename, "class": img_class})

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"results": results}


@app.get("/history/{image_dir}", tags=["History"])
def image_history(image_dir: str):
    """

    """
    files = os.listdir(image_dir)
    history = [file for file in files]
    return {"history": history}


@app.get("/img/{image_dir}/{image_name}", tags=["History"])
def get_classified_image(image_dir: str, image_name: str):
    """

    """
    file_path = f"{image_dir}/{image_name}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    uvicorn.run("app.fake_face_detection_api:app", host="0.0.0.0", port=8000, reload=True)

