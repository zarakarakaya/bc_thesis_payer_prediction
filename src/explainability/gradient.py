
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data import load_data, PlayerDataset

from src.config import load_config


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

folder = "focal_loss_2"

out_dir = Path("results") / folder / "model.pt"
model = torch.load(out_dir, weights_only=False)
cfg = load_config("configs/best.yaml")
X, y, feature_names  = load_data(cfg.data.path)

ds = PlayerDataset(X, y)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


loader = DataLoader(
    ds,
    batch_size=64,
    num_workers=0,
    pin_memory=pin_memory,
    shuffle=True
)
model.eval()

all_importance = []

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


indices = np.argsort(final_importance)[::-1]

sorted_importance = final_importance[indices]
sorted_features = feature_names[indices]


plt.figure(figsize=(14, 6))
plt.bar(range(len(sorted_importance)), sorted_importance)
plt.xticks(range(len(sorted_importance)), sorted_features, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Gradient Feature Importance ")
plt.tight_layout()


output_path = Path("results") / folder / "gradient_importance_all.png"
plt.savefig(output_path)

plt.show()
