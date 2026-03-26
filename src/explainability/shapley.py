import shap

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data import load_data, PlayerDataset

from src.config import load_config

import numpy as np
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
def model_forward(x):
    return torch.sigmoid(model(x)).squeeze(1)
from sklearn.metrics import precision_recall_curve

folder = "focal_loss_2"

out_dir = Path("results") / folder / "model.pt"


cfg = load_config("configs/best.yaml")
X, y, feature_names = load_data(cfg.data.path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

model = torch.load(out_dir, weights_only=False)

model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()


thresholds = np.linspace(0.01, 0.99, 99)
best_f1, best_t = -1.0, 0.5

for t in thresholds:
    preds = (probs >= t).astype(int)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t




y_pred = (probs >= best_t).astype(int)
groups = {
    "TP":  (y_pred == 1) & (y_test == 1),
    "FP": (y_pred == 1) & (y_test == 0),
    "TN": (y_pred == 0) & (y_test == 0),
    "FN": (y_pred == 0) & (y_test == 1)
}
bg_original = shap.sample(X_train, 100)
explainer = shap.DeepExplainer(model, bg_original)

for title, mask in groups.items():
    indices = np.where(mask)[0]

    sample_idx = indices[:100]
    X_sub = X_test[sample_idx]
    
    shap_vals = explainer.shap_values(X_sub)

    shap_vals = np.squeeze(shap_vals)

    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP: {title}")
    shap.summary_plot(shap_vals, X_sub.cpu().numpy(), feature_names=feature_names, show=False)
    plt.show()