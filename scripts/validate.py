import argparse
import torch
from torch.utils.data import DataLoader

from src.data import load_data, PlayerDataset
from src.model import MLP
from src.trainer import Trainer
from src.kfold import make_folds
from src.config import load_config

def main(args):
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y = load_data(cfg.data.path)

    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    pos_weight = cfg.training.pos_weight
    n_folds = cfg.training.n_folds

    for fold, (train_idx, val_idx) in enumerate(
        make_folds(X, y, n_splits=n_folds)
    ):
        print(f"\n===== Fold {fold + 1}/{n_folds} =====")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = PlayerDataset(X_train, y_train)
        val_ds = PlayerDataset(X_val, y_val)

        pin_memory = (device.type == "cuda")
        
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
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

        trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
