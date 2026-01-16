import wandb
from src.config import load_config
from scripts.train import run_training


def sweep_train():
    with wandb.init():
        cfg = load_config("configs/base.yaml")

        for k, v in wandb.config.items():
            if hasattr(cfg.training, k):
                setattr(cfg.training, k, v)
            elif hasattr(cfg.model, k):
                setattr(cfg.model, k, v)

        run_training(cfg, use_wandb=True)


if __name__ == "__main__":
    sweep_train()