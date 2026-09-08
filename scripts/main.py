
from src.data import load_data
from src.config import load_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from scripts.validate import validate
from scripts.train import run_training
import json
import wandb
def train(cfg= None,  use_wandb = False):
    folder = "work"
    type = "sampler"
    if not cfg:
        cfg = load_config("configs/best.yaml")
    X, y, feature_names  = load_data(cfg.data.path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    avg = validate(cfg, type = type, X = X_train, y = y_train, use_wandb=False)

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",  # matches your imbalanced payer/non-payer split
        n_jobs=-1,
    )
    selector = SelectFromModel(rf, threshold="median")  # keeps top ~half by importance
    selector.fit(X_train, y_train)

    selected_mask = selector.get_support()
    selected_feature_names = [f for f, keep in zip(feature_names, selected_mask) if keep]
    print(f"selected {len(selected_feature_names)} / {len(feature_names)} features:")
    print(selected_feature_names)

    X_train = selector.transform(X_train)   
    X_test = selector.transform(X_test) 

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_test  = scaler.transform(X_test)
    data =  (X_train, X_test, y_train, y_test)
    history, trainer = run_training(cfg, type = type, data=data, log=True, use_wandb=use_wandb, folder=folder)

    out_dir = Path("results") / folder
    if use_wandb:
        out_dir = out_dir / wandb.run.id
    trainer.save_model(out_dir/"model.pt")
    with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=4)
    with open(out_dir / "val_history.json", "w") as f:
            json.dump(avg, f, indent=4)

if __name__ == "__main__":
    train()