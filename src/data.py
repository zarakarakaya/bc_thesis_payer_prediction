import pandas as pd
import torch
from torch.utils.data import Dataset


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



def load_data(path, encode = True):
    df = pd.read_parquet(path)
    df = df.set_index('player_id')

 
    df = pd.get_dummies(df, columns=['ecpm_network', 'media_source', 'platform'])

    y = df["payer"].values.astype("float32")

    X_df = df.drop(columns=["revenue_d28", "payer"])
    
    if encode:
        X = X_df.values.astype("float32")
    else:
        X = X_df.values
    feature_names = X_df.columns
    return X, y, feature_names

