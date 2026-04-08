import multiprocessing as mp
import wandb

from src.utils import load_yaml
from scripts.sweep import sweep_train


def main():
    sweep_config = load_yaml("configs/sweep_pos_weight.yaml")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="bc_final_bce_logit_loss_vol2",
        entity="karakayazara-comenius-university-in-bratislava",
    )

    wandb.agent(
        sweep_id,
        function=sweep_train,
        count=None,
    )


if __name__ == "__main__":
    mp.freeze_support()   
    main()
