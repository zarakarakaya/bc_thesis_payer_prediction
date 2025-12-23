import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def make_folds(
    X,
    y,
    n_splits: int,
    seed: int = 42
):

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    splits = splitter.split(X, y)

    for train_idx, val_idx in splits:
        yield train_idx, val_idx
