
from pathlib import Path


from src.explainability.helper import get_samples
import json
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
folder = "bce"

out_dir = Path("results") / folder /"best"

val_history_file = out_dir / "val_history.json"
history_file = out_dir / "history.json"
matrix_file = out_dir / "confusion_matrix.txt"
#k fold training loss vs validation loss

with open(val_history_file, "r") as f:
    val_history = json.load(f)

train_loss = val_history.get("loss", [])
val_loss = val_history.get("val_loss", [])

plt.figure()
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig(out_dir / "loss_curve.png")
plt.show()

from src.config import load_config
#confusion matrix



cm = np.loadtxt(matrix_file, delimiter=",", dtype=int)
df = pd.DataFrame(cm)
perc = df.div(df.sum(axis=1), axis=0)
names = np.array([['True Neg', 'False Pos'], ['False Neg', 'True Pos']])
annot = (
        names + "\n" + 
        df.astype(str) + "\n" + 
        perc.multiply(100).round(2).astype(str) + "%"
    )
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    perc, 
    annot=annot, 
    fmt='', 
    cmap="flare", 
    linewidths=1, 
    cbar=True,
    annot_kws={"fontsize": 10}
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(out_dir / "confusion_matrix.png")
plt.show()

# precision and recall 
with open(history_file, "r") as f:
    history = json.load(f)

precision = history.get("val_precision", [])
recall = history.get("val_recall", [])

plt.figure()
plt.plot(precision, label="Precision")
plt.plot(recall, label="Recall")
plt.xlabel("Epoch")
plt.title("Precision and Recall")
plt.legend()
plt.grid()
plt.savefig(out_dir / "precision_recall.png")
plt.show()

