import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.models import load_model

# ================================
# CONFIG
# ================================
MODEL_PATH = "model/model.keras"
CSV_PATH   = "C:/Users/fazee/Documents/dataset/labels.csv"
IMAGE_DIR  = "C:/Users/fazee/Documents/dataset/images/"
IMG_SIZE   = (160, 160)
BATCH_SIZE = 32

# ================================
# LOAD CSV
# ================================
df = pd.read_csv(CSV_PATH)
df["image_path"] = df["filename"].apply(lambda f: os.path.join(IMAGE_DIR, f))
df = df.rename(columns={"label_id": "label_idx"})

# ================================
# LOAD MODEL
# ================================
print("📂 Loading model...")
model = load_model(MODEL_PATH)

# ================================
# DATA LOADER
# ================================
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

paths = df["image_path"].values
labels = df["label_idx"].values

test_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
test_ds = test_ds.map(load_image).batch(BATCH_SIZE)

# ================================
# MODEL PREDICTIONS
# ================================
y_true = []
y_pred = []

for imgs, lbls in test_ds:
    preds = model.predict(imgs)
    preds = np.argmax(preds, axis=1)
    
    y_true.extend(lbls.numpy().tolist())
    y_pred.extend(preds.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ================================
# METRICS CALCULATION
# ================================
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted")
rec  = recall_score(y_true, y_pred, average="weighted")
f1   = f1_score(y_true, y_pred, average="weighted")

report = classification_report(y_true, y_pred, digits=4)
cm = confusion_matrix(y_true, y_pred)

# ================================
# SAVE METRICS TO TXT
# ================================
with open("diagnostics.txt", "w", encoding="utf-8") as f:
    f.write("==== MODEL PERFORMANCE METRICS ====\n\n")
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall:    {rec:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n\n")
    
    f.write("==== CLASSIFICATION REPORT ====\n")
    f.write(report + "\n\n")
    
    f.write("==== CONFUSION MATRIX ====\n")
    for row in cm:
        f.write(" ".join(str(x) for x in row) + "\n")

print("✅ Metrics saved to diagnostics.txt")
