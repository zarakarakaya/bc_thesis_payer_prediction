
from pathlib import Path


import json
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.trainer import Trainer
import torch

from sklearn.model_selection import train_test_split
from src.data import load_data

from sklearn.preprocessing import StandardScaler

from pathlib import Path

from torch.utils.data import DataLoader
from src.config import load_config
from src.data import load_data, PlayerDataset

SMALL_FIG = (3.3, 2.6)  
FONT_SIZE = 9
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "lines.linewidth": 1.2,

})

folder = "sampler"

out_dir = Path("results") / folder /"best"

val_history_file = out_dir / "val_history.json"
history_file = out_dir / "history.json"
matrix_file = out_dir / "confusion_matrix.txt"

#k fold training loss vs validation loss

with open(val_history_file, "r") as f:
    val_history = json.load(f)

train_loss = val_history.get("loss", [])
val_loss = val_history.get("val_loss", [])

plt.figure(figsize=SMALL_FIG)
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir / "loss_curve.png",dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()


#confusion matrix


cm = np.loadtxt(matrix_file, delimiter=",", dtype=int)
cm = cm[::-1, ::-1] 

df = pd.DataFrame(cm)
perc = df.div(df.sum(axis=1), axis=0)
names = np.array([['True Pos', 'False Neg'], ['False Pos', 'True Neg']]) 
annot = (
        names + "\n" + 
        df.astype(str) + "\n" + 
        perc.multiply(100).round(2).astype(str) + "%"
    )
fig, ax = plt.subplots(figsize=SMALL_FIG)
sns.heatmap(
    perc, 
    annot=annot, 
    fmt='', 
    cmap="Blues", 
    linewidths=1, 
    cbar=True,
    annot_kws={"fontsize": 10}
)
ax.set_xticklabels(['1', '0'])
ax.set_yticklabels(['1', '0'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(out_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()

# precision and recall 
with open(history_file, "r") as f:
    history = json.load(f)

precision = history.get("val_precision", [])
recall = history.get("val_recall", [])

plt.figure(figsize=SMALL_FIG)
plt.plot(precision, label="Precision")
plt.plot(recall, label="Recall")
plt.xlabel("Epoch")
plt.title("Precision and Recall")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir / "precision_recall.png",dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(out_dir / "model.pt", weights_only=False)
model = model.to(device)
model.eval()
trainer = Trainer(model=model, optimizer=None, criterion=None, scheduler=None, device=device)

cfg = load_config("configs/best.yaml")
X, y, feature_names = load_data(cfg.data.path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

val_ds = PlayerDataset(X_test, y_test)
val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)

# ROC
fpr, tpr, _, roc_auc = trainer.roc_values(val_loader)
plt.figure(figsize=SMALL_FIG)
plt.plot(fpr, tpr)
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
plt.tight_layout()
plt.savefig(out_dir / "roc_curve.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()

# PR
precision_vals, recall_vals, _, pr_auc = trainer.pr_values(val_loader)
baseline = np.mean(y_test)
plt.figure(figsize=SMALL_FIG)
plt.plot(recall_vals, precision_vals, label=f"AUC = {pr_auc:.2f}")
plt.hlines(baseline, 0, 1, linestyles="dashed", label="Baseline")
plt.fill_between(recall_vals, precision_vals, alpha=0.3)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "pr_curve.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
