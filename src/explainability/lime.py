import lime
import lime.lime_tabular
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn
from src.data import load_data
from src.config import load_config
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import pandas as pd

from src.explainability.helper import get_samples

folder = "focal_loss"
out_dir = Path("results") / folder / "best" / "model.pt"
cfg = load_config("configs/best.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(out_dir, weights_only=False)
model.to(device)
model.eval()

#nacitavam data
df = pd.read_parquet(cfg.data.path)
df['actions_per_session'] = df['total_actions'] / df['session_count']
df = df[df['payer_d0'] == 0].drop(columns='payer_d0')
X = df.drop(columns=["payer_d7"])
y = df["payer_d7"].values.astype("float32")



#feature names PRED one hot
feature_names = list(X.columns)

categorical_features = []
categorical_names = {}

col = "network_name"
idx = X.columns.get_loc(col)

categorical_features.append(idx)

#lime potrebuje categorical features encodnute
le = LabelEncoder()
X[col] = le.fit_transform(X[col])
categorical_names[idx] = list(le.classes_)

X = X.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#feature names PO one hot + potrebujem fitnut scaler na original datach
X_helper, y_helper, model_feature_names = load_data(cfg.data.path, encode=True)
model_feature_names = list(model_feature_names)

X_helper_train, _, _, _ = train_test_split(X_helper, y_helper, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
scaler.fit(X_helper_train)

#lime potrebuje tuto funkciu, ktora zoberie tie pertrubed data a da ich do modelu
#lenze model je nauceny na one hot takze to musime upravit
def predict_fn(x):

    df_x = pd.DataFrame(x, columns=feature_names)
    df_x[col] = le.inverse_transform(df_x[col].astype(int))
    df_x = pd.get_dummies(df_x, columns=['network_name'])
    df_x = df_x.reindex(columns=model_feature_names, fill_value=0)
    
    df_x = df_x.values.astype(np.float32)
    x_scaled = scaler.transform(df_x)
    x_tensor = torch.tensor(x_scaled).to(device)
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    return np.concatenate([1 - probs, probs], axis=1)

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=["non-payer", "payer"],
    categorical_features=categorical_features,
    categorical_names=categorical_names,
    mode="classification"
)


import matplotlib.pyplot as plt
groups = get_samples(cfg, model)

for group, mask in groups.items():

    indices = np.where(mask)[0]

    idx = indices[0]

    exp = explainer.explain_instance(
        X_test[idx], 
        predict_fn, 
        num_features=10
    )


    fig = exp.as_pyplot_figure()
    plt.title(f"LIME :{group}")
    plt.tight_layout()
    output_path = Path("results") / folder / "exp" / "lime" / f"{group}_lime.png"
    plt.savefig(output_path)
    plt.show()