
from src.data import load_data
from src.config import load_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from scripts.validate import validate
from scripts.train import run_training
import json
import wandb

from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

from src.artifacts import save_run_bundle
from src.data import load_data
from src.preprocessing import Preprocessor

def make_run_dir(use_wandb):
    if use_wandb:
        run_id = wandb.run.id
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path("results") / "work" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir

def train(cfg=None, use_wandb=False):
    TYPE = 'sampler'  # 'focal_loss', 'sampler', or 'bce'
    if cfg is None:
        cfg = load_config("configs/best.yaml")

    run_dir = make_run_dir(use_wandb)

    X, y, feature_names, player_ids = load_data(
        cfg.data.path
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        ids_train,
        ids_val,
    ) = train_test_split(
        X,
        y,
        player_ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    splits = pd.DataFrame({
        "player_id": pd.concat(
            [
                ids_train.reset_index(drop=True),
                ids_val.reset_index(drop=True),
            ],
            ignore_index=True,
        ),
        "split": (
            ["train"] * len(ids_train)
            + ["validation"] * len(ids_val)
        ),
    })
        # K-fold experiment
    cv_history = validate(
        cfg,
        type=TYPE,
        X=X_train,
        y=y_train,
        use_wandb=False,
    )

    # Final preprocessing
    preprocessor = Preprocessor()

    X_train = preprocessor.fit_transform(
        X_train,
        y_train,
        feature_names,
    )

    X_val = preprocessor.transform(X_val)

    history, trainer = run_training(
        cfg,
        type=TYPE,
        data=(X_train, X_val, y_train, y_val),
        use_wandb=use_wandb,
    )


    final_threshold = history["val_threshold"][-1]



    save_run_bundle(
        run_dir=run_dir,
        trainer=trainer,
        preprocessor=preprocessor,
        feature_columns=preprocessor.feature_columns,
        threshold=final_threshold,
        cfg=cfg,
        history=history,
        cv_history=cv_history,
        splits=splits,
    )

if __name__ == "__main__":
    train()