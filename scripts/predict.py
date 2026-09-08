"""
scripts/predict.py

Loads the deployment bundle (model.pt, scaler.pkl, feature_columns.json,
threshold.json) from results/work/best/ and scores new players.

Run from the repo root:
    python -m scripts.predict --input path/to/players.parquet --output predictions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.model import MLP  # noqa: F401 - required for torch.load to resolve the pickled class

BEST_DIR = Path("results") / "best"


def load_bundle(bundle_dir: Path = BEST_DIR):
    model_path = bundle_dir / "model.pt"
    scaler_path = bundle_dir / "scaler.pkl"
    columns_path = bundle_dir / "feature_columns.json"
    threshold_path = bundle_dir / "threshold.json"

    for p in (model_path, scaler_path, columns_path, threshold_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {p}. Run extract_artifacts.py "
                f"first if this bundle predates the artifact-saving changes."
            )

    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()

    scaler = joblib.load(scaler_path)

    with open(columns_path) as f:
        feature_columns = json.load(f)

    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    return model, scaler, feature_columns, threshold


def load_input(input_path: Path) -> pd.DataFrame:
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        return pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input file type: {input_path.suffix}")


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Encodes network_name exactly as src.data.load_data does, then
    reindexes to the exact training column set/order. Unseen categories ->
    all-zero row for that dummy; missing columns -> filled with 0; extra
    columns -> dropped."""
    encoded = pd.get_dummies(df, columns=["network_name"])
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded.values.astype("float32")


def run_inference(df: pd.DataFrame, bundle_dir: Path = BEST_DIR) -> pd.DataFrame:
    model, scaler, feature_columns, threshold = load_bundle(bundle_dir)

    if "player_id" not in df.columns:
        raise ValueError("Input file must contain a 'player_id' column")

    player_ids = df["player_id"].values
    feature_df = df.drop(columns=["player_id"], errors="ignore")

    X = prepare_features(feature_df, feature_columns)

    expected_dim = len(feature_columns)
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Feature count mismatch after reindex: got {X.shape[1]}, "
            f"model expects {expected_dim}. Check feature_columns.json "
            f"matches this bundle's model.pt."
        )

    X_scaled = scaler.transform(X)

    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy().flatten()

    preds = (probs >= threshold).astype(int)

    return pd.DataFrame({
        "player_id": player_ids,
        "predicted_payer_prob": probs,
        "predicted_payer": preds,
    })


INPUT_PATH = Path("data") /  "data_test_23.parquet"
OUTPUT_PATH = Path("results") /  "predictions_23.csv"


def main():
    df = load_input(INPUT_PATH)
    print(f"loaded {len(df)} players from {INPUT_PATH}")

    results_df = run_inference(df, BEST_DIR)
    print(results_df.head())
    print(f"predicted payers: {results_df['predicted_payer'].sum()} / {len(results_df)}")

    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"predictions written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()