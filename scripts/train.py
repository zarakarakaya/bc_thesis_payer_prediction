import argparse
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

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
from src.utils import save_pr, save_roc
def run_training(cfg, type, use_wandb=False, log= False, data = None, folder = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0

    print(f"Using device: {device}")
    if data:
         X_train, X_test, y_train, y_test = data
    else:
        X, y, _ = load_data(cfg.data.path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 
        X_test  = scaler.transform(X_test)

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
    
    

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_dataloader,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
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
    if log:
        threshold = history["best_t"][-1]
        import time
        out_dir = Path("results") / folder
        if use_wandb:
            out_dir = out_dir / wandb.run.id
        else:
            run_name = f"run_{int(time.time())}"
            out_dir = out_dir/ run_name

        out_dir.mkdir(parents=True, exist_ok=True)
        
        cm = trainer.confusion_matrix(val_loader, threshold = threshold)
        
        cm_path = out_dir / "confusion_matrix.txt"

        
        np.savetxt(cm_path, cm, delimiter=",", fmt='%d')

        cfg_path = out_dir / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(namespace_to_dict(cfg), f, indent=2)

        roc_path = out_dir / "roc_curve.png"
        save_roc(trainer, val_loader, roc_path)

        pr_path = out_dir / "pr_curve.png"
        save_pr(trainer, val_loader, pr_path, y_test)


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
   
    if use_wandb:
        for epoch, (loss, acc, f1, t, precision, recall) in enumerate(
            zip(history["loss"], history["val_acc"], history["val_f1"], history["best_t"], history["val_precision"], history["val_recall"])
        ):
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_accuracy": acc,
                    "val_f1": f1,
                    "best_t": t,
                    "val_precision": precision, 
                    "val_recall": recall
                }
            )
    return history, trainer
