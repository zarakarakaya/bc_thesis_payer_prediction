import yaml
from pathlib import Path
import json
import numpy as np
from src.trainer import Trainer
import matplotlib.pyplot as plt

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def save_roc(trainer, data_loader, path):
    fpr, tpr, thresholds, roc_auc = trainer.roc_values(data_loader)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.fill_between(fpr, tpr, alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.savefig(path)
    plt.close()

def save_pr(trainer, data_loader, path, y_test):

    precision, recall, thresholds, pr_auc = trainer.pr_values(data_loader)

    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    baseline = np.mean(y_test)
    plt.hlines(baseline, 0, 1, linestyles="dashed", label="Baseline")
    plt.fill_between(recall, precision, alpha=0.3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(path)
    plt.close()