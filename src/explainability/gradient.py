
import torch
from torch.utils.data import DataLoader

from src.data import load_data, PlayerDataset

from src.config import load_config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from src.explainability.helper import get_samples

folder = "bce"
mode = "group"
out_dir = Path("results") / folder /"best"/ "model.pt"
model = torch.load(out_dir, weights_only=False)

cfg = load_config("configs/best.yaml")
X, y, feature_names  = load_data(cfg.data.path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test  = scaler.transform(X_test)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
pin_memory = (device.type == "cuda")

model.eval()

groups = get_samples(cfg, model)

group_importances = {}

for group, mask in groups.items():
    indices = np.where(mask)[0]
    if mode == "single": 
        sample_idx = indices[0]
        inputs = torch.tensor(X_test[[sample_idx]], dtype=torch.float32).to(device)
        inputs.requires_grad = True

        outputs = model(inputs)
        probs = torch.sigmoid(outputs).squeeze()

        model.zero_grad()
        probs.backward()

        final_importance = inputs.grad.abs().squeeze().cpu().numpy()
    else:
        all_importance = []

        
        X_group = X_test[indices]
        y_group = y_test[indices]
        group_ds = PlayerDataset(X_group, y_group)

        loader = DataLoader(
            group_ds,
            batch_size=64,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=False
        )
        
    
        for inputs, _ in loader:
            inputs = inputs.to(device)
            inputs.requires_grad = True

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze()

            model.zero_grad()
            probs.sum().backward()

            grads = inputs.grad

            importance = grads.abs().mean(dim=0)

            all_importance.append(importance.detach().cpu())

        final_importance = torch.stack(all_importance).mean(dim=0).numpy()
    group_importances[group] = final_importance
for group, importance in group_importances.items():
    group_importances[group] = importance / importance.sum()       
importance = np.sum(list(group_importances.values()), axis=0)

top_indices = np.argsort(importance)[::-1]
top_features = feature_names[top_indices]


n_groups = len(group_importances)
x = np.arange(len(top_features))
width = 0.8 / n_groups


plt.figure(figsize=(14, 5))
for i, (group, importance) in enumerate(group_importances.items()):
    vals = importance[top_indices]
    plt.bar(x + i * width, vals, width, label=group)

plt.xticks(x + width * (n_groups - 1) / 2, top_features, rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Gradient feature importance by group")
plt.legend()
plt.tight_layout()

output_path = Path("results") / folder / "exp" / "gradient" / f"{mode}_all_groups_gradient.png"
plt.savefig(output_path)
plt.show()
