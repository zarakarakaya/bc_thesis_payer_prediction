import argparse
import torch
from torch.utils.data import DataLoader

from src.data import load_data, TabularDataset
from src.model import MLP
from src.trainer import Trainer
from src.kfold import make_folds


def main(args):

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # data
    X, y = load_data(args.data_path)

    # k-fold
    for fold, (train_idx, val_idx) in enumerate(
        make_folds(X, y, n_splits=args.n_folds)
    ):
        print(f"\n===== Fold {fold + 1}/{args.n_folds} =====")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TabularDataset(X_train, y_train)
        val_ds = TabularDataset(X_val, y_val)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # model
        model = MLP(
            input_dim=X.shape[1],
            hidden_layers=[128, 64],
            activations=["relu", "relu"],
            dropout=0.1,
        ).to(device)

        # training setup
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr
        )

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.pos_weight], device=device)
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

        # train
        trainer.fit(
            train_loader,
            val_loader,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=70.0)
    parser.add_argument("--n-folds", type=int, default=5)

    args = parser.parse_args()
    main(args)
