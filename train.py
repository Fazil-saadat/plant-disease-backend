import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



# ==============================
# CONFIG
# ==============================
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
CSV_PATH = "dataset/labels.csv"
IMAGES_DIR = "dataset/images"

# ==============================
# LOAD METADATA
# ==============================
labels_df = pd.read_csv(CSV_PATH)

# Ensure class_id is integer encoded
labels_df['label_id'] = labels_df['label_id'].values.astype(np.int32)

class_names = labels_df['label'].unique().tolist()
num_classes = len(class_names)
print("Classes:", class_names)

# Train/val/test split
train_df, temp_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['label_id'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_id'], random_state=42)

# ==============================
# DATA LOADER
# ==============================
def load_image(filename, label):
    # convert tensor -> string
    filename = filename.numpy().decode("utf-8")
    img_path = os.path.join(IMAGES_DIR, filename)

    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # scale [-1,1]
    return img, label


def dataframe_to_dataset(df, shuffle=True):
    filenames = df['filename'].values.astype(str)
    labels = df['label_id'].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    def wrapper(filename, label):
        img, lbl = tf.py_function(
            load_image,
            [filename, label],
            [tf.float32, tf.int32]
        )
        img.set_shape(IMG_SIZE + (3,))  # static shape
        lbl.set_shape([])  # scalar
        return img, lbl

    ds = ds.map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = dataframe_to_dataset(train_df, shuffle=True)
val_ds = dataframe_to_dataset(val_df, shuffle=False)
test_ds = dataframe_to_dataset(test_df, shuffle=False)

# ==============================
# VERIFY DATA ALIGNMENT
# ==============================
for imgs, lbls in train_ds.take(1):
    for i in range(5):
        plt.imshow((imgs[i].numpy() + 1) / 2)  # back to [0,1] for display
        plt.title(f"Label: {class_names[lbls[i].numpy()]}")
        plt.axis("off")
        plt.show()

# ==============================
# MODEL
# ==============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base initially

global_avg = tf.keras.layers.GlobalAveragePooling2D()
dropout = tf.keras.layers.Dropout(0.2)
prediction_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = global_avg(x)
x = dropout(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

# ==============================
# TRAIN (Phase 1: frozen base)
# ==============================
initial_epochs = 10
history = model.fit(train_ds, epochs=initial_epochs, validation_data=val_ds)

# ==============================
# FINE-TUNE (Phase 2: unfreeze last 20 layers)
# ==============================
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_ds
)

# ==============================
# EVALUATE
# ==============================
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# Get ground truth labels and predictions
y_true = []
y_pred = []

for imgs, labels in test_ds:
    preds = model.predict(imgs)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ---------------- Per-class Metrics ----------------
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

model.save("model.keras")