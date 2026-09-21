import numpy as np
import wandb

from src.data import load_data
from src.kfold import make_folds
from src.preprocessing import Preprocessor
from scripts.train import run_training


def average_history(fold_history):
    keys = [
        "loss",
        "val_loss",
        "val_acc",
        "val_f1",
        "val_threshold",
        "val_precision",
        "val_recall",
    ]

    avg = {}

    for key in keys:
        arr = np.array(
            [history[key] for history in fold_history],
            dtype=float,
        )

        avg[key] = arr.mean(axis=0).tolist()

    return avg


def validate(
    cfg,
    type,
    X=None,
    y=None,
    feature_names=None,
    use_wandb=False,
):
    # Allows validate() to also be called independently.
    if X is None or y is None or feature_names is None:
        (
            X,
            y,
            feature_names,
            _,
        ) = load_data(cfg.data.path)

    n_folds = cfg.training.n_folds
    fold_history = []

    if use_wandb:
        wandb.config.update(
            {
                "hidden_layers": cfg.model.hidden_layers,
                "activations": cfg.model.activations,
                "hidden_layers_str": "-".join(
                    map(str, cfg.model.hidden_layers)
                ),
                "activations_str": "-".join(
                    map(str, cfg.model.activations)
                ),
            },
            allow_val_change=True,
        )

    for fold, (train_idx, val_idx) in enumerate(
        make_folds(
            X,
            y,
            n_splits=n_folds,
        )
    ):
        print(
            f"\n===== Fold "
            f"{fold + 1}/{n_folds} ====="
        )

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Each fold gets its own fitted preprocessor.
        # For now this is just StandardScaler.
        preprocessor = Preprocessor()

        X_train = preprocessor.fit_transform(
            X_train,
            y_train,
            feature_names,
        )

        X_val = preprocessor.transform(
            X_val
        )

        data = (
            X_train,
            X_val,
            y_train,
            y_val,
        )

        history, _ = run_training(
            cfg,
            type,
            data=data,
            use_wandb=use_wandb,
        )

        fold_history.append(
            history
        )

    avg = average_history(
        fold_history
    )

    if use_wandb:
        for epoch, (
            loss,
            val_loss,
            acc,
            f1,
            threshold,
            precision,
            recall,
        ) in enumerate(
            zip(
                avg["loss"],
                avg["val_loss"],
                avg["val_acc"],
                avg["val_f1"],
                avg["val_threshold"],
                avg["val_precision"],
                avg["val_recall"],
            )
        ):
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_loss": val_loss,
                    "val_accuracy": acc,
                    "val_f1": f1,
                    "val_threshold": threshold,
                    "val_precision": precision,
                    "val_recall": recall,
                }
            )

    return avg