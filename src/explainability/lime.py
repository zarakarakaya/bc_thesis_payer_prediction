import lime
import lime.lime_tabular
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn
from src.data import load_data
from src.config import load_config

import numpy as np
from pathlib import Path
import pandas as pd

from helper import get_samples

folder = "focal_loss_2"
out_dir = Path("results") / folder / "model.pt"
cfg = load_config("configs/best.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(out_dir, weights_only=False)
model.to(device)
model.eval()


df = pd.read_parquet(cfg.data.path)
df['actions_per_session'] = df['total_actions'] / df['session_count']
df = df[df['payer_d0'] == 0].drop(columns='payer_d0')
X_df = df.drop(columns=["payer_d7"])
y = df["payer_d7"].values.astype("float32")
feature_names = list(X_df.columns)

X = X_df.copy()

categorical_features = []
categorical_names = {}

col = "network_name"
idx = X_df.columns.get_loc(col)

categorical_features.append(idx)

le = LabelEncoder()
X[col] = le.fit_transform(X[col])
categorical_names[idx] = list(le.classes_)

X = X.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
_, _, model_feature_names = load_data(cfg.data.path, encode=True)
model_feature_names = list(model_feature_names)
def predict_fn(x):
# 1. Convert LIME's numpy (9 cols) to DataFrame
    df_x = pd.DataFrame(x, columns=feature_names)
    
    # 2. Decode the LabelEncoded network column back to strings
    df_x[col] = le.inverse_transform(df_x[col].astype(int))
    
    # 3. One-hot encode (this creates multiple columns)
    df_x = pd.get_dummies(df_x, columns=['network_name'])
    
    # 4. Reindex to the 22 columns your model expects
    # IMPORTANT: Ensure model_feature_names is defined outside this function
    df_x = df_x.reindex(columns=model_feature_names, fill_value=0)
    
    # 5. FORCE conversion to float32 to fix the "object_" error
    # This converts True/False from dummies into 1.0/0.0
    final_numpy_array = df_x.values.astype(np.float32)
    
    # 6. Pass to model
    x_tensor = torch.tensor(final_numpy_array).to(device)
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
        
    return np.concatenate([1 - probs, probs], axis=1)

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=["non-payer", "payer"],
    categorical_features=categorical_features,
    categorical_names=categorical_names,
    mode="classification"
)
random_idx = np.random.randint(0, len(X_test))
print(f"Explaining random sample at index: {random_idx}")

exp = explainer.explain_instance(
    X_test[random_idx], 
    predict_fn, 
    num_features=10
)

import matplotlib.pyplot as plt
groups = get_samples(cfg, model)

# 3. Iterate through each category
for group_name, mask in groups.items():
    # Convert the boolean mask to integer indices
    matching_indices = np.where(mask)[0]
    
    if len(matching_indices) == 0:
        print(f"\n--- No samples found for {group_name} ---")
        continue
        
    # Take the first available sample in this group
    idx = matching_indices[0]
    print(f"\nExplaining {group_name} (Index: {idx})")

    # Generate the explanation
    exp = explainer.explain_instance(
        X_test[idx], 
        predict_fn, 
        num_features=10
    )

    # Plotting
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation: {group_name} (Sample {idx})")
    plt.tight_layout()
    plt.show()