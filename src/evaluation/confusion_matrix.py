from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Paths
INPUT_FILE = Path("data/evaluation/back_dev_locate_prep.csv")
OUTPUT_DIR = Path("data/evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)


# ============================================================
# CONFUSION MATRIX: PHOTO TYPE
# ============================================================

typed = df.dropna(subset=["true_type", "photo_type"])

labels = sorted(
    set(typed["true_type"].unique())
    | set(typed["photo_type"].unique())
)

cm = confusion_matrix(
    typed["true_type"],
    typed["photo_type"],
    labels=labels,
)

# Save confusion matrix as CSV
cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

cm_df.index.name = "true_type"

csv_file = OUTPUT_DIR / "photo_type_confusion_matrix.csv"
cm_df.to_csv(csv_file)

print(f"Confusion matrix CSV saved to: {csv_file}")


# Save confusion matrix as PNG
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)

plt.title("Photo Type Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

image_file = OUTPUT_DIR / "photo_type_confusion_matrix.png"
plt.savefig(image_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Confusion matrix image saved to: {image_file}")

