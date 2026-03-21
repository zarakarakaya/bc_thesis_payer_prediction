
from src.data import load_data
from src.config import load_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scripts.validate import validate
from scripts.train import run_training
import json
def train(cfg= None,  use_wandb = False):
    folder = "focal_loss"
    type = "focal_loss"
    if not cfg:
        cfg = load_config("configs/best.yaml")
    X, y = load_data(cfg.data.path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    avg = validate(cfg, type = type, X = X_train, y = y_train, use_wandb=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_test  = scaler.transform(X_test)
    data =  (X_train, X_test, y_train, y_test)
    history, trainer = run_training(cfg, type = type, data=data, log=True, use_wandb=use_wandb, folder=folder)
    
    out_dir = Path("results") / folder
    trainer.save_model(out_dir)
    with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=4)
    with open(out_dir / "val_history.json", "w") as f:
            json.dump(avg, f, indent=4)