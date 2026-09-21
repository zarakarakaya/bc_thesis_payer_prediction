import pandas as pd
import torch
from torch.utils.data import Dataset


CATEGORICAL_COLUMNS = [
    "ecpm_network",
    "media_source",
    "platform",
]

TARGET_COLUMN = "payer"

NON_FEATURE_COLUMNS = [
    "player_id",
    "payer",
    "revenue",
]


class PlayerDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx]),
        )

def prepare_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.drop(
        columns=NON_FEATURE_COLUMNS,
        errors="ignore",
    )

    return encode_features(feature_df)

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical = [
        c for c in CATEGORICAL_COLUMNS
        if c in df.columns
    ]

    return pd.get_dummies(
        df,
        columns=categorical,
        dtype=float,
    )


def load_data(path):
    df = pd.read_parquet(path)

    player_ids = df["player_id"].copy()

    y = df[TARGET_COLUMN].to_numpy(dtype="float32")

    feature_df = prepare_feature_df(df)
    
    X = feature_df.to_numpy(dtype="float32")

    return (
        X,
        y,
        feature_df.columns.tolist(),
        player_ids,
    )