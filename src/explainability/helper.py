import torch
from src.data import load_data
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import StandardScaler
def get_samples(cfg, model, scale = True):
    X, y, feature_names = load_data(cfg.data.path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if scale: 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 
        X_test  = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)



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
    return groups