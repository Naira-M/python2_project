import pickle

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from source.utils import load_model, classify_image


MODEL = load_model("files/checkpoint.pth")

app = FastAPI(
    title="Fake Face Image Detection API",
    description="Takes uploaded image and returns its class (fake or real).",
    version="0.0.1",
    openapi_tags=[{"name": "Segment", "description": "API endpoints related to image segmentation"}]
)


# TODO: There might be something about model in root
@app.get("/", tags=["Greeting"])
def root():
    """Greet a user."""
    return {"message": "Hello World"}


@app.get("/metadata", tags=["Model"])
def model_metadata():
    """
    Metadata of the latest trained mode.
    """
    with open("files/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return JSONResponse(content=metadata)


@app.post("/classify/", tags=["Classify"])
def classify(
        image: UploadFile = File(description="A required image file for classification.")):
    """
    Finds out if the uploaded image is real or generated (fake).
    """
    if not image.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

    try:
        img_class = classify_image(MODEL, image.file)
        return JSONResponse(content={
            "filename": image.filename,
            "class": img_class
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.fake_face_detection_api:app", host="0.0.0.0", port=8000, reload=True)

