from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

IMG_SIZE = 160
MODEL_PATH = "model/model.keras"
CLASSES_JSON = "metadata/classes.json"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load classes and treatments
with open(CLASSES_JSON, "r") as f:
    classes_dict = json.load(f)

app = FastAPI(title="Plant Disease Classifier")

# Mount static directories
app.mount("/css", StaticFiles(directory="static/css"), name="css")
app.mount("/js", StaticFiles(directory="static/js"), name="js")
app.mount("/images", StaticFiles(directory="static/images"), name="images")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Utility
def preprocess_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(os.path.dirname(__file__), "templates/home.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        img_array = preprocess_image(file_path)
        preds = model.predict(img_array)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds)) * 100

        class_info = classes_dict.get(str(pred_idx), {})
        pred_class = class_info.get("class_name", "Unknown")
        treatment = class_info.get("treatment", "Not available")

        return JSONResponse(
            content={
                "prediction": pred_class,
                "treatment": treatment,
                "confidence": round(confidence,2),
                "uploaded_image": f"/uploads/{file.filename}"
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
