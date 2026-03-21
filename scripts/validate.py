import numpy as np
from src.data import load_data
from src.kfold import make_folds
from scripts.train import run_training
import wandb
import argparse
from sklearn.preprocessing import StandardScaler
from src.config import load_config
def average_history(fold_history):

    keys = ["loss", "val_loss", "val_acc", "val_f1", "best_t", "val_precision", "val_recall"]
    avg = {}

    for k in keys:
        arr = np.array([h[k] for h in fold_history], dtype=float) 
        avg[k] = arr.mean(axis=0).tolist()

    return avg
def validate(cfg, type,  X=None, y=None, use_wandb = False):

    if X is None or y is None:
        X, y = load_data(cfg.data.path)


    n_folds = cfg.training.n_folds
    fold_history = []

    if use_wandb:
        wandb.config.update(
            {
                "hidden_layers": cfg.model.hidden_layers,
                "activations": cfg.model.activations,
                "hidden_layers_str": "-".join(map(str, cfg.model.hidden_layers)),
                "activations_str": "-".join(map(str, cfg.model.activations)),
            },
            allow_val_change=True,
        )
    for fold, (train_idx, val_idx) in enumerate(
        make_folds(X, y, n_splits=n_folds)
    ):
        print(f"\n===== Fold {fold + 1}/{n_folds} =====")


        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 
        X_val   = scaler.transform(X_val)
        data =  (X_train, X_val, y_train, y_val)
        history, _ = run_training(cfg, type, use_wandb=use_wandb, data=data)
        fold_history.append(history)

    avg = average_history(fold_history)


    if use_wandb:
        for epoch, (loss, val_loss, acc, f1, t, precision, recall) in enumerate(
            zip(avg["loss"], 
                avg["val_loss"],
                avg["val_acc"], 
                avg["val_f1"], 
                avg["best_t"], 
                avg["val_precision"], 
                avg["val_recall"]
            )
        ):
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_loss": val_loss,
                    "val_accuracy": acc,
                    "val_f1": f1,
                    "best_t": t,
                    "val_precision": precision, 
                    "val_recall": recall

                }
            )

    return avg
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate(cfg, use_wandb=False)

if __name__ == "__main__":
    main()
