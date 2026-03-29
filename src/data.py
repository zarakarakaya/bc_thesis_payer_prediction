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
    #df = df.set_index('player_id')

    df['actions_per_session'] = df['total_actions'] / df['session_count']
    df = df[df['payer_d0'] == 0].drop(columns='payer_d0')
    if encode:
        df = pd.get_dummies(df, columns=['network_name'])

    X_df = df.drop(columns=["payer_d7"])
    y = df["payer_d7"].values.astype("float32")
    if encode:
        X = X_df.values.astype("float32")
    else:
        X = X_df.values 
    feature_names = X_df.columns
    return X, y, feature_names


