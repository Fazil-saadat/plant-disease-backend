import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ---------------- FASTAPI SETUP ----------------
app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates (HTML)
templates = Jinja2Templates(directory="templates")

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "model/model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (must match training order)
class_names = [
    "Apple___Scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Grape___Black_rot",
    "Grape___Esca",
    "Grape___Leaf_blight",
    "Grape___healthy"
]

IMG_SIZE = (160, 160)


# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    return {
        "class": class_names[pred_class],
        "confidence": round(confidence, 4)
    }


# ---------------- RUN ----------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
