
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

folder = "focal_loss"
mode = "singleee"
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


    indices = np.argsort(final_importance)[::-1]

    sorted_importance = final_importance[indices]
    sorted_features = feature_names[indices]


    plt.figure(figsize=(14, 6))
    plt.bar(range(len(sorted_importance)), sorted_importance)
    plt.xticks(range(len(sorted_importance)), sorted_features, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(f"Gradient feature importance: {group}")
    plt.tight_layout()


    output_path = Path("results") / folder / "exp" / "gradient" / f"{mode}_{group}_gradient.png"
    plt.savefig(output_path)

    plt.show()

