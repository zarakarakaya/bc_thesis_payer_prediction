from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from torch.utils.data import DataLoader

from src.artifacts import load_model
from src.data import prepare_feature_df, PlayerDataset
from src.trainer import Trainer


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

RUN_DIR = (
    Path("results")
    / "work"
    / "20260910_142004"
)

# This must match the name used when splits.parquet was created in main.py.
# Your current main.py uses "validation".
EVAL_SPLIT_NAME = "validation"

SMALL_FIG = (3.3, 2.6)
FONT_SIZE = 9

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "lines.linewidth": 1.2,
})


# ---------------------------------------------------------------------
# Load saved run
# ---------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
print(f"Loading run: {RUN_DIR}")


model = load_model(
    RUN_DIR,
    device=device,
)


preprocessor = joblib.load(
    RUN_DIR / "preprocessor.pkl"
)


with open(RUN_DIR / "config.json", "r") as f:
    run_config = json.load(f)


with open(RUN_DIR / "threshold.json", "r") as f:
    threshold = float(
        json.load(f)["threshold"]
    )


with open(RUN_DIR / "history.json", "r") as f:
    history = json.load(f)


with open(RUN_DIR / "cv_history.json", "r") as f:
    cv_history = json.load(f)


splits = pd.read_parquet(
    RUN_DIR / "splits.parquet"
)


print(f"Saved threshold: {threshold:.4f}")


# ---------------------------------------------------------------------
# Load exact data used by this run
# ---------------------------------------------------------------------

data_path = run_config["data"]["path"]

raw_df = pd.read_parquet(data_path)
print(
    "raw rows:",
    len(raw_df),
)

print(
    "raw unique players:",
    raw_df["player_id"].nunique(),
)

print(
    "raw duplicate rows by player:",
    raw_df["player_id"].duplicated().sum(),
)

print(
    "split rows:",
    len(splits),
)

print(
    "split unique players:",
    splits["player_id"].nunique(),
)

cross_split = (
    splits
    .groupby("player_id")["split"]
    .nunique()
)

print(
    "players appearing in multiple splits:",
    (cross_split > 1).sum(),
)

eval_ids = (
    splits
    .loc[
        splits["split"] == EVAL_SPLIT_NAME,
        ["player_id"],
    ]
)


if eval_ids.empty:
    raise ValueError(
        f"No rows with split='{EVAL_SPLIT_NAME}' "
        f"found in {RUN_DIR / 'splits.parquet'}"
    )


# Merge rather than isin() so the saved split determines the cohort.
eval_df = eval_ids.merge(
    raw_df,
    on="player_id",
    how="left",
    validate="one_to_one",
)


if eval_df.isna().all(axis=1).any():
    raise ValueError(
        "Some saved player IDs could not be found "
        "in the current dataset."
    )


print(f"Evaluation players: {len(eval_df):,}")


# ---------------------------------------------------------------------
# Recreate features exactly as training expects
# ---------------------------------------------------------------------

X_eval_df = prepare_feature_df(eval_df)


# The selector was fitted using the complete encoded training feature
# set in this exact order. New/missing dummy columns therefore need
# to be aligned before calling preprocessor.transform().
X_eval_df = X_eval_df.reindex(
    columns=preprocessor.input_feature_names,
    fill_value=0,
)


X_eval = preprocessor.transform(
    X_eval_df.to_numpy(dtype="float32")
)


y_eval = eval_df["payer"].to_numpy(
    dtype="float32"
)


print(
    f"Model features: {X_eval.shape[1]} | "
    f"Positive players: {int(y_eval.sum()):,} / {len(y_eval):,}"
)


# ---------------------------------------------------------------------
# DataLoader + Trainer
# ---------------------------------------------------------------------

eval_ds = PlayerDataset(
    X_eval,
    y_eval,
)


eval_loader = DataLoader(
    eval_ds,
    batch_size=run_config["training"]["batch_size"],
    shuffle=False,
)


trainer = Trainer(
    model=model,
    optimizer=None,
    criterion=None,
    scheduler=None,
    device=device,
)


# ---------------------------------------------------------------------
# Final-model training / validation loss
# ---------------------------------------------------------------------

train_loss = history.get("loss", [])
val_loss = history.get("val_loss", [])


plt.figure(figsize=SMALL_FIG)

plt.plot(
    train_loss,
    label="Training Loss",
)

plt.plot(
    val_loss,
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    RUN_DIR / "loss_curve.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)

plt.show()


# ---------------------------------------------------------------------
# Final-model F1
# ---------------------------------------------------------------------

val_f1 = history.get("val_f1", [])


if val_f1:
    plt.figure(figsize=SMALL_FIG)

    plt.plot(
        val_f1,
        label="Validation F1",
    )

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Validation F1")
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        RUN_DIR / "f1_curve.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.show()


# ---------------------------------------------------------------------
# Threshold over epochs
# ---------------------------------------------------------------------

val_threshold = history.get(
    "val_threshold",
    [],
)


if val_threshold:
    plt.figure(figsize=SMALL_FIG)

    plt.plot(
        val_threshold,
        label="Optimal Threshold",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Threshold")
    plt.title("Validation Threshold")
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        RUN_DIR / "threshold_curve.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.show()


# ---------------------------------------------------------------------
# Precision and recall over epochs
# ---------------------------------------------------------------------

precision_history = history.get(
    "val_precision",
    [],
)

recall_history = history.get(
    "val_recall",
    [],
)


plt.figure(figsize=SMALL_FIG)

plt.plot(
    precision_history,
    label="Precision",
)

plt.plot(
    recall_history,
    label="Recall",
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision and Recall")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    RUN_DIR / "precision_recall_history.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)

plt.show()


# ---------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------

cm = trainer.confusion_matrix(
    eval_loader,
    threshold=threshold,
)


# sklearn gives:
#
# [[TN, FP],
#  [FN, TP]]
#
# Reorder for display as:
#
# [[TP, FN],
#  [FP, TN]]
cm_display = cm[::-1, ::-1]


cm_df = pd.DataFrame(cm_display)


row_sums = cm_df.sum(axis=1).replace(0, 1)

cm_perc = cm_df.div(
    row_sums,
    axis=0,
)


names = np.array([
    ["True Pos", "False Neg"],
    ["False Pos", "True Neg"],
])


annot = (
    names
    + "\n"
    + cm_df.astype(str)
    + "\n"
    + cm_perc
        .multiply(100)
        .round(2)
        .astype(str)
    + "%"
)


fig, ax = plt.subplots(
    figsize=SMALL_FIG
)

sns.heatmap(
    cm_perc,
    annot=annot,
    fmt="",
    cmap="Blues",
    linewidths=1,
    cbar=True,
    annot_kws={"fontsize": 10},
    ax=ax,
)

ax.set_xticklabels(["1", "0"])
ax.set_yticklabels(["1", "0"])

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(
    f"Confusion Matrix (t={threshold:.2f})"
)

plt.tight_layout()

plt.savefig(
    RUN_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)

plt.show()


# Also save numeric confusion matrix
np.savetxt(
    RUN_DIR / "confusion_matrix.txt",
    cm,
    delimiter=",",
    fmt="%d",
)


# ---------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------

fpr, tpr, _, roc_auc = trainer.roc_values(
    eval_loader
)


plt.figure(figsize=SMALL_FIG)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}",
)

plt.fill_between(
    fpr,
    tpr,
    alpha=0.3,
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    RUN_DIR / "roc_curve.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)

plt.show()


# ---------------------------------------------------------------------
# Precision-recall curve
# ---------------------------------------------------------------------

(
    precision_vals,
    recall_vals,
    _,
    pr_auc,
) = trainer.pr_values(
    eval_loader
)


baseline = float(
    np.mean(y_eval)
)


plt.figure(figsize=SMALL_FIG)

plt.plot(
    recall_vals,
    precision_vals,
    label=f"AUC = {pr_auc:.4f}",
)

plt.hlines(
    baseline,
    0,
    1,
    linestyles="dashed",
    label=f"Baseline = {baseline:.4f}",
)

plt.fill_between(
    recall_vals,
    precision_vals,
    alpha=0.3,
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    RUN_DIR / "pr_curve.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)

plt.show()


# ---------------------------------------------------------------------
# CV history
# ---------------------------------------------------------------------

cv_train_loss = cv_history.get(
    "loss",
    [],
)

cv_val_loss = cv_history.get(
    "val_loss",
    [],
)


if cv_train_loss and cv_val_loss:
    plt.figure(figsize=SMALL_FIG)

    plt.plot(
        cv_train_loss,
        label="CV Training Loss",
    )

    plt.plot(
        cv_val_loss,
        label="CV Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("K-Fold Average Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        RUN_DIR / "cv_loss_curve.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.show()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

tn, fp, fn, tp = cm.ravel()


final_f1 = (
    history["val_f1"][-1]
    if history.get("val_f1")
    else None
)

final_precision = (
    history["val_precision"][-1]
    if history.get("val_precision")
    else None
)

final_recall = (
    history["val_recall"][-1]
    if history.get("val_recall")
    else None
)


print("\n=== Final model ===")

if final_f1 is not None:
    print(
        f"F1:        {final_f1:.4f}"
    )

if final_precision is not None:
    print(
        f"Precision: {final_precision:.4f}"
    )

if final_recall is not None:
    print(
        f"Recall:    {final_recall:.4f}"
    )

print(
    f"Threshold: {threshold:.4f}"
)

print(
    f"ROC AUC:   {roc_auc:.4f}"
)

print(
    f"PR AUC:    {pr_auc:.4f}"
)

print(
    f"Baseline:  {baseline:.4f}"
)

print("\nConfusion matrix:")
print(f"TP: {tp:,}")
print(f"FP: {fp:,}")
print(f"FN: {fn:,}")
print(f"TN: {tn:,}")