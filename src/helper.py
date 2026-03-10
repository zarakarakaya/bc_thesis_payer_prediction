import torch
from data import load_data
import pandas as pd
def main():
    # path to your dataset (adjust if needed)
    data_path = "data\data.parquet"  # or directly your data file path
    df = pd.read_parquet(data_path)
    

    # load data
    X, y = load_data(data_path)

    # if y is a torch tensor
    if isinstance(y, torch.Tensor):
        num_payers = (y == 1).sum().item()
        total = y.shape[0]
    else:
        # if y is numpy array
        import numpy as np
        num_payers = np.sum(y == 1)
        total = len(y)

    print(f"Total samples: {total}")
    print(f"Number of payers (payer_d7 = 1): {num_payers}")
    print(f"Payer ratio: {num_payers / total:.4f}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.head(5))

if __name__ == "__main__":
    main()