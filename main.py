import base64
from io import BytesIO

import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from model import ImagePreprocessor, OnnxModel

app = FastAPI(title="MTailor Assessment API")

MODEL_PATH = "models/model.onnx"
model = OnnxModel(MODEL_PATH)
preprocessor = ImagePreprocessor()


class ImageRequest(BaseModel):
    image_data: str


class PredictionResponse(BaseModel):
    prediction: int
    status_code: int


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    image_bytes = base64.b64decode(request.image_data)
    image = Image.open(BytesIO(image_bytes))

    prediction = model.predict_with_preprocessing(image, preprocessor)
    prediction_value = int(np.argmax(prediction[0]))

    return {"prediction": prediction_value, "status_code": 200}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
