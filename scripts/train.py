import argparse
import torch
from torch.utils.data import DataLoader

from src.data import load_data, PlayerDataset
from src.model import MLP
from src.trainer import Trainer
from src.config import load_config,namespace_to_dict
from sklearn.model_selection import train_test_split
import wandb

from pathlib import Path
import json
import numpy as np

def run_training(cfg, use_wandb=False, log= False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        num_workers = 0
    else:
        num_workers = 4
    print(f"Using device: {device}")
    if use_wandb:
        wandb.config.update(
            {
                "hidden_layers": cfg.model.hidden_layers,
                "activations": cfg.model.activations,
            },
            allow_val_change=True,
        )
    X, y = load_data(cfg.data.path)

    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    pos_weight = cfg.training.pos_weight

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_ds = PlayerDataset(X_train, y_train)
    val_ds = PlayerDataset(X_test, y_test)

    pin_memory = (device.type == "cuda")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    model = MLP(
        input_dim=X.shape[1],
        hidden_layers=cfg.model.hidden_layers,
        activations=cfg.model.activations
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.95
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
    )

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        verbose=True,
    )
    if log:
        cm = trainer.confusion_matrix(val_loader)

        out_dir = Path("results") / wandb.run.id if use_wandb else Path("results") / "no_wandb"
        out_dir.mkdir(parents=True, exist_ok=True)

        cm_path = out_dir / "confusion_matrix.txt"
        with open(cm_path, "w") as f:
            f.write("Confusion Matrix (rows=true, cols=pred)\n\n")
            f.write(np.array2string(cm))

        cfg_path = out_dir / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(namespace_to_dict(cfg), f, indent=2)

    if use_wandb:
        for epoch, (loss, acc, f1) in enumerate(
            zip(history["loss"], history["val_acc"],  history["val_f1"])
        ):
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_accuracy": acc,
                    "val_f1": f1,
                }
            )

    return history



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg, use_wandb=False)


if __name__ == "__main__":
    main()
