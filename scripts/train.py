import argparse
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from src.data import load_data, PlayerDataset
from src.model import MLP
from src.trainer import Trainer
from src.criterion import FocalLoss
from src.config import load_config, namespace_to_dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from src.runtime import configure_threads
from src.utils import save_pr, save_roc


def make_loader(
    dataset,
    batch_size,
    shuffle=False,
    sampler=None,
    num_workers=0,
    pin_memory=False,
):
    """DataLoader that fetches a whole batch per __getitem__ call.

    Wrapping the sampler in a BatchSampler and passing batch_size=None turns
    off DataLoader's automatic batching: instead of calling __getitem__ once
    per row and collating 128 tiny tensors, it hands the dataset the whole
    index list and gets a ready batch back. The sampling order is unchanged.
    """
    if sampler is None:
        sampler = (
            RandomSampler(dataset)
            if shuffle
            else SequentialSampler(dataset)
        )

    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False,
    )

    return DataLoader(
        dataset,
        batch_size=None,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
def run_training(cfg, type, data, use_wandb=False ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0

    n_threads = configure_threads(device)

    print(f"Using device: {device} ({n_threads} threads)")

    X_train, X_test, y_train, y_test = data


    if type == "focal_loss":
        sampler = None
        shuffle_dataloader = True

        alpha = cfg.training.alpha
        gamma = cfg.training.gamma

        criterion = FocalLoss(alpha=alpha, gamma=gamma)

    elif type == "sampler":
        shuffle_dataloader = False

        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        pos_weight = 1
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

    elif type == "bce":  
        sampler = None
        shuffle_dataloader = True

        pos_weight = cfg.training.pos_weight
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    
    



    train_ds = PlayerDataset(X_train, y_train)
    val_ds = PlayerDataset(X_test, y_test)

    pin_memory = (device.type == "cuda")
    
    

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_dataloader,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_layers=cfg.model.hidden_layers,
        activations=cfg.model.activations
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
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
   

        for epoch, (loss, acc, f1, t, precision, recall) in enumerate(
            zip(history["loss"], history["val_acc"], history["val_f1"], history["val_threshold"], history["val_precision"], history["val_recall"])
        ):
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_accuracy": acc,
                    "val_f1": f1,
                    "val_threshold": t,
                    "val_precision": precision, 
                    "val_recall": recall
                }
            )
    return history, trainer
