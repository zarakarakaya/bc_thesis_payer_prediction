import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.artifacts import load_model
from src.data import prepare_feature_df


INPUT_PATH = Path("data") / "bc_inference_input_ver11.parquet"
OUTPUT_PATH = Path("results") / "bc_inference_input_ver11.csv"

RUN_DIR = (
    Path("results")
    / "work"
    / "20260910_142004"
)


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:

    encoded = prepare_feature_df(df)

    # feature_columns are the selected features the final model
    # was trained on, in their exact training order.
    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return encoded.to_numpy(dtype="float32")


def load_bundle(bundle_dir: Path = RUN_DIR):

    scaler_path = bundle_dir / "scaler.pkl"
    columns_path = bundle_dir / "feature_columns.json"
    threshold_path = bundle_dir / "threshold.json"

    required_files = [
        bundle_dir / "model.pt",
        scaler_path,
        columns_path,
        threshold_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    model = load_model(
        bundle_dir,
        device="cpu",
    )

    scaler = joblib.load(
        scaler_path
    )

    with open(columns_path, "r") as f:
        feature_columns = json.load(f)

    with open(threshold_path, "r") as f:
        threshold = float(
            json.load(f)["threshold"]
        )

    return (
        model,
        scaler,
        feature_columns,
        threshold,
    )


def load_input(
    input_path: Path,
) -> pd.DataFrame:

    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)

    if input_path.suffix == ".csv":
        return pd.read_csv(input_path)

    raise ValueError(
        f"Unsupported input file type: "
        f"{input_path.suffix}"
    )


def run_inference(
    df: pd.DataFrame,
    bundle_dir: Path = RUN_DIR,
) -> pd.DataFrame:

    (
        model,
        scaler,
        feature_columns,
        threshold,
    ) = load_bundle(bundle_dir)

    if "player_id" not in df.columns:
        raise ValueError(
            "Input file must contain a "
            "'player_id' column"
        )

    player_ids = df["player_id"].to_numpy()

    X = prepare_features(
        df,
        feature_columns,
    )

    expected_dim = len(feature_columns)

    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Feature count mismatch: "
            f"got {X.shape[1]}, "
            f"expected {expected_dim}"
        )

    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(X_tensor)

        probs = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
            .flatten()
        )

    print(f"threshold: {threshold:.4f}")

    print(
        "probability percentiles:",
        np.percentile(
            probs,
            [50, 75, 90, 95, 97, 99, 99.5, 99.9],
        ),
    )

    print(
        f"min probability: {probs.min():.4f}"
    )

    print(
        f"max probability: {probs.max():.4f}"
    )

    preds = (
        probs >= threshold
    ).astype(int)

    preds = (
        probs >= threshold
    ).astype(int)

    return pd.DataFrame({
        "player_id": player_ids,
        "predicted_payer_prob": probs,
        "predicted_payer": preds,
    })


def main():

    df = load_input(INPUT_PATH)

    print(
        f"loaded {len(df)} players "
        f"from {INPUT_PATH}"
    )
    print(
        df[
            [
                "session_count",
                "time_played",
                "jobs_completed",
                "ecpm",
                "max_level",
            ]
        ].describe()
    )
    results_df = run_inference(
        df,
        RUN_DIR,
    )

    print(results_df.head())

    print(
        f"predicted payers: "
        f"{results_df['predicted_payer'].sum()} "
        f"/ {len(results_df)}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"predictions written to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()