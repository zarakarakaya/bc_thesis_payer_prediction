import json

from pathlib import Path

import joblib

import torch

from src.config import namespace_to_dict
from src.model import MLP


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        
def load_saved_config(run_dir):
    with open(Path(run_dir) / "config.json") as f:
        return json.load(f)



def save_run_bundle(
    run_dir,
    trainer,
    preprocessor,
    feature_columns,
    threshold,
    cfg,
    history,
    cv_history,
    splits,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": trainer.model.state_dict(),
        "input_dim": len(feature_columns),
        "hidden_layers": list(cfg.model.hidden_layers),
        "activations": list(cfg.model.activations),
    }

    torch.save(
        checkpoint,
        run_dir / "model.pt",
    )

    joblib.dump(
        preprocessor,
        run_dir / "preprocessor.pkl",
    )

    joblib.dump(
        preprocessor.scaler,
        run_dir / "scaler.pkl",
    )

    save_json(
        list(feature_columns),
        run_dir / "feature_columns.json",
    )

    save_json(
        {"threshold": float(threshold)},
        run_dir / "threshold.json",
    )

    save_json(
        namespace_to_dict(cfg),
        run_dir / "config.json",
    )

    save_json(
        history,
        run_dir / "history.json",
    )

    save_json(
        cv_history,
        run_dir / "cv_history.json",
    )

    splits.to_parquet(
        run_dir / "splits.parquet",
        index=False,
    )

def load_model(run_dir, device="cpu"):
    run_dir = Path(run_dir)

    checkpoint = torch.load(
        run_dir / "model.pt",
        map_location=device,
        weights_only=True,
    )

    model = MLP(
        input_dim=checkpoint["input_dim"],
        hidden_layers=checkpoint["hidden_layers"],
        activations=checkpoint["activations"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model