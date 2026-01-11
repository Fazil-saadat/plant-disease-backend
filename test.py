import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ================================
# 1) PATHS
# ================================
CSV_PATH = "C:/Users/fazee/Documents/dataset/labels.csv"
IMAGES_DIR = "C:/Users/fazee/Documents/dataset/images"
MODEL_PATH = "model/model.keras"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================
# 2) Load labels file
# ================================
df = pd.read_csv(CSV_PATH)
df["label_id"] = df["label_id"].astype(np.int32)

# ================================
# 3) Extract label names
# ================================
class_names = df.drop_duplicates("label_id").sort_values("label_id")["label"].tolist()
num_classes = len(class_names)

# ================================
# 4) Load trained model
# ================================
model = tf.keras.models.load_model(MODEL_PATH)

# ================================
# 5) Preprocessing
# ================================
IMG_SIZE = (160, 160)

def preprocess_image(filename):
    img_path = os.path.join(IMAGES_DIR, filename)
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = img / 255.0
    return img

# ================================
# 6) Build dataset arrays
# ================================
images = []
labels = []

for _, row in df.iterrows():
    img = preprocess_image(row["filename"])
    images.append(img)
    labels.append(row["label_id"])

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# ================================
# 7) OVERALL MODEL TESTING (Complete Dataset)
# ================================
print("\n" + "="*50)
print("OVERALL MODEL TESTING (Complete Dataset)")
print("="*50)

# Evaluate on entire dataset
loss, acc = model.evaluate(images, labels, verbose=1)
print(f"Overall Test Accuracy: {acc:.4f}")
print(f"Overall Test Loss: {loss:.4f}")

# Get predictions for entire dataset
predictions = model.predict(images, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Overall metrics
overall_precision = precision_score(labels, y_pred, average='weighted')
overall_recall = recall_score(labels, y_pred, average='weighted')
overall_f1 = f1_score(labels, y_pred, average='weighted')

print(f"\nOverall Metrics (Weighted):")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall: {overall_recall:.4f}")
print(f"F1-Score: {overall_f1:.4f}")

# Generate classification report
class_report = classification_report(labels, y_pred, target_names=class_names)
print(f"\nClassification Report:\n{class_report}")

# Save classification report to file
with open(f"{RESULTS_DIR}/overall_classification_report.txt", "w") as f:
    f.write("OVERALL MODEL EVALUATION\n")
    f.write("="*50 + "\n\n")
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"Test Loss: {loss:.4f}\n\n")
    f.write("Overall Metrics (Weighted):\n")
    f.write(f"Precision: {overall_precision:.4f}\n")
    f.write(f"Recall: {overall_recall:.4f}\n")
    f.write(f"F1-Score: {overall_f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)

# Overall confusion matrix
overall_cm = confusion_matrix(labels, y_pred)
overall_cm_norm = confusion_matrix(labels, y_pred, normalize='true')

# Save overall confusion matrix data
overall_cm_json = json.dumps(overall_cm.tolist())
overall_cm_norm_json = json.dumps(overall_cm_norm.tolist())

# Create overall results dictionary
overall_results = {
    "Test_Accuracy": acc,
    "Test_Loss": loss,
    "Precision_weighted": overall_precision,
    "Recall_weighted": overall_recall,
    "F1_weighted": overall_f1,
    "CM_Counts_Values": overall_cm_json,
    "CM_Normalized_Values": overall_cm_norm_json
}

# Save overall results to CSV
overall_df = pd.DataFrame([overall_results])
overall_df.to_csv(f"{RESULTS_DIR}/overall_test_results.csv", index=False)
print(f"\nSaved overall results to: {RESULTS_DIR}/overall_test_results.csv")

# ================================
# 8) Plot Overall Confusion Matrices
# ================================
# Counts confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(overall_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Overall Confusion Matrix (Counts)\nComplete Dataset Testing", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/overall_confusion_matrix_counts.png", dpi=300, bbox_inches='tight')
plt.close()

# Normalized confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(overall_cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Overall Confusion Matrix (Normalized)\nComplete Dataset Testing", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/overall_confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# 9) Per-class Accuracy Bar Chart
# ================================
# Calculate per-class accuracy from confusion matrix
per_class_accuracy = overall_cm_norm.diagonal()

plt.figure(figsize=(max(10, num_classes), 6))
bars = plt.bar(range(num_classes), per_class_accuracy)
plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
plt.title("Per-Class Accuracy (Overall Testing)", fontsize=14, fontweight='bold')
plt.xlabel("Class", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add accuracy values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.2f}", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/overall_per_class_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()

# ================================
# 10) 5-Fold Testing (Your original code)
# ================================
print("\n" + "="*50)
print("5-FOLD CROSS-VALIDATION TESTING")
print("="*50)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = {
    "Fold": [],
    "Accuracy": [],
    "Precision_micro": [],
    "Recall_micro": [],
    "F1_micro": [],
    "CM_Counts_Values": [],
    "CM_Normalized_Values": []
}

fold = 1

for train_idx, test_idx in kf.split(images):
    X_test = images[test_idx]
    y_test = labels[test_idx]

    # Evaluate accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {fold} - Accuracy: {acc:.4f}")

    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Metrics
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    # Confusion matrices
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    # Convert matrices to JSON string
    cm_json = json.dumps(cm.tolist())
    cm_norm_json = json.dumps(cm_norm.tolist())

    # Save fold results in dictionary
    fold_results["Fold"].append(fold)
    fold_results["Accuracy"].append(acc)
    fold_results["Precision_micro"].append(precision)
    fold_results["Recall_micro"].append(recall)
    fold_results["F1_micro"].append(f1)
    fold_results["CM_Counts_Values"].append(cm_json)
    fold_results["CM_Normalized_Values"].append(cm_norm_json)

    # Save confusion matrix plots
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (Counts) - Fold {fold}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_counts_fold{fold}.png")
    plt.close()

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (Normalized) - Fold {fold}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_normalized_fold{fold}.png")
    plt.close()

    fold += 1

# ================================
# 11) Save 5-fold results in ONE CSV
# ================================
results_df = pd.DataFrame(fold_results)

# Calculate mean values
mean_accuracy = results_df["Accuracy"].mean()
mean_precision = results_df["Precision_micro"].mean()
mean_recall = results_df["Recall_micro"].mean()
mean_f1 = results_df["F1_micro"].mean()

mean_row = {
    "Fold": "Mean",
    "Accuracy": mean_accuracy,
    "Precision_micro": mean_precision,
    "Recall_micro": mean_recall,
    "F1_micro": mean_f1,
    "CM_Counts_Values": "",
    "CM_Normalized_Values": ""
}

results_df.loc[len(results_df)] = mean_row

results_df.to_csv(f"{RESULTS_DIR}/5fold_results_complete.csv", index=False)

print(f"\n5-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"\nSaved: {RESULTS_DIR}/5fold_results_complete.csv")

# ================================
# 12) Save bar chart of accuracy (5-fold)
# ================================
plt.figure(figsize=(9,6))
bars = plt.bar(results_df["Fold"][:-1], results_df["Accuracy"][:-1])

# Label bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.3f}", ha='center', fontsize=11, fontweight='bold')

plt.title("5-Fold Cross-Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/5fold_accuracy_bar_chart.png")
plt.close()

print(f"Saved: {RESULTS_DIR}/5fold_accuracy_bar_chart.png")

# ================================
# 13) Create Comprehensive Summary Report
# ================================
summary_report = f"""
MODEL EVALUATION SUMMARY REPORT
{"="*60}

DATASET INFORMATION:
- Total samples: {len(df)}
- Number of classes: {num_classes}
- Classes: {', '.join(class_names)}

OVERALL TESTING (Complete Dataset):
{"-"*40}
- Test Accuracy: {overall_results['Test_Accuracy']:.4f}
- Test Loss: {overall_results['Test_Loss']:.4f}
- Precision (Weighted): {overall_results['Precision_weighted']:.4f}
- Recall (Weighted): {overall_results['Recall_weighted']:.4f}
- F1-Score (Weighted): {overall_results['F1_weighted']:.4f}

5-FOLD CROSS-VALIDATION RESULTS:
{"-"*40}
- Mean Accuracy: {mean_accuracy:.4f}
- Mean Precision: {mean_precision:.4f}
- Mean Recall: {mean_recall:.4f}
- Mean F1-Score: {mean_f1:.4f}

PERFORMANCE COMPARISON:
{"-"*40}
Overall Test Accuracy: {overall_results['Test_Accuracy']:.4f}
5-Fold Mean Accuracy: {mean_accuracy:.4f}
Difference: {overall_results['Test_Accuracy'] - mean_accuracy:.4f}

FILES GENERATED:
{"-"*40}
1. overall_test_results.csv - Overall evaluation metrics
2. overall_confusion_matrix_counts.png - Overall confusion matrix (counts)
3. overall_confusion_matrix_normalized.png - Overall confusion matrix (normalized)
4. overall_per_class_accuracy.png - Per-class accuracy bar chart
5. overall_classification_report.txt - Detailed classification report
6. 5fold_results_complete.csv - 5-fold cross-validation results
7. 5fold_accuracy_bar_chart.png - 5-fold accuracy visualization
8. confusion_matrix_counts_foldX.png - Per-fold confusion matrices
9. confusion_matrix_normalized_foldX.png - Per-fold normalized matrices
"""

# Save summary report
with open(f"{RESULTS_DIR}/evaluation_summary.txt", "w") as f:
    f.write(summary_report)

print("\n" + "="*50)
print("EVALUATION COMPLETE")
print("="*50)
print(summary_report)
print(f"\nAll results saved in: {RESULTS_DIR}/")
print("="*50)