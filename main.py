from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from disease_data import DISEASE_DATABASE
from PIL import Image
import tensorflow as tf
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
    "Cherry___Powdery_mildew", "Cherry___Healthy", 
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___Healthy"
]

# Disease database with multi-language support


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    disease_name: str
    description: str
    symptoms: str
    treatment: str
    prevention: str
    language: str

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))
    image = np.array(image)
    return image

@app.get("/")
async def root():
    return {"message": "Plant Disease Classification API"}

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    language: str = "en"  # Default to English
):
    # Validate language
    if language not in ["en", "fa", "ps"]:
        language = "en"  # Default to English if invalid language
    
    # Read and process image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Make prediction
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    # Get disease information in requested language
    disease_info = DISEASE_DATABASE.get(predicted_class, {}).get(language, {})
    
    # If disease not found in database, provide default response
    if not disease_info:
        disease_info = {
            "disease_name": predicted_class.replace("___", " ").replace("_", " "),
            "description": f"No detailed information available for {predicted_class}",
            "symptoms": "Symptoms information not available",
            "treatment": "Treatment information not available",
            "prevention": "Prevention information not available"
        }
    
    return PredictionResponse(
        prediction=predicted_class,
        confidence=float(confidence),
        disease_name=disease_info["disease_name"],
        description=disease_info["description"],
        symptoms=disease_info["symptoms"],
        treatment=disease_info["treatment"],
        prevention=disease_info["prevention"],
        language=language
    )

@app.get("/diseases")
async def get_diseases(language: str = "en"):
    """Get all available diseases information in specified language"""
    if language not in ["en", "fa", "ps"]:
        language = "en"
    
    diseases_info = {}
    for class_name in CLASS_NAMES:
        disease_info = DISEASE_DATABASE.get(class_name, {}).get(language, {})
        if disease_info:
            diseases_info[class_name] = disease_info
    
    return {"language": language, "diseases": diseases_info}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)