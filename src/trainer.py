import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

import numpy as np

def best_f1_threshold(probs, labels, thresholds):
    """Max F1 over a threshold grid, computed in one pass.

    Equivalent to looping over the grid and calling f1_score for each
    threshold (zero_division=0), but sorts once and uses cumulative counts
    instead of rescanning every row 99 times.
    """
    order = np.argsort(probs, kind="stable")
    sorted_probs = probs[order]
    sorted_labels = labels[order]

    # positives among rows with prob strictly below each threshold
    cum_pos = np.cumsum(sorted_labels)
    total_pos = cum_pos[-1] if cum_pos.size else 0.0

    below = np.searchsorted(sorted_probs, thresholds, side="left")
    pos_below = np.where(below > 0, cum_pos[np.maximum(below - 1, 0)], 0.0)

    tp = total_pos - pos_below
    predicted_pos = probs.size - below

    denominator = predicted_pos + total_pos
    f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator > 0,
    )

    best = int(np.argmax(f1))

    return float(f1[best]), float(thresholds[best])


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        device,
        use_wandb=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.use_wandb = use_wandb

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach() * x.size(0)

        return float(total_loss) / len(train_loader.dataset)

    def eval_epoch(self, val_loader):
        self.model.eval()
        all_probs, all_labels = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = torch.sigmoid(logits)
                loss = self.criterion(logits, y)
                total_val_loss += loss.detach() * x.size(0)
                all_probs.append(probs.cpu()) 
                all_labels.append(y.cpu())

        all_probs = torch.cat(all_probs).numpy().flatten()
        all_labels = torch.cat(all_labels).numpy().flatten()

        avg_val_loss = float(total_val_loss) / len(val_loader.dataset)
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1, val_threshold = best_f1_threshold(
            all_probs,
            all_labels,
            thresholds,
        )

        best_preds = (all_probs >= val_threshold).astype(int)
        acc = accuracy_score(all_labels, best_preds)
        precision = precision_score(all_labels, best_preds, zero_division=0)
        recall = recall_score(all_labels, best_preds, zero_division=0)
        return avg_val_loss, acc, best_f1, val_threshold, precision, recall

    def fit(self, train_loader, val_loader, epochs, verbose=False):
        history = {
            "loss": [],
            "val_loss": [], 
            "val_acc": [], 
            "val_f1": [], 
            "val_threshold": [], 
            "val_precision": [], 
            "val_recall": [],
            }

        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            val_loss, acc, f1, val_threshold, precision, recall = self.eval_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()


            history["loss"].append(loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(acc)
            history["val_f1"].append(f1)
            history["val_threshold"].append(val_threshold)
            history["val_precision"].append(precision)
            history["val_recall"].append(recall)
            if verbose:
                print(f"Epoch {epoch+1} | loss={loss:.4f} | val loss={val_loss:.4f} | val_acc={acc:.4f} | val_f1={f1:.4f}")

        return history

    def confusion_matrix(self, val_loader, threshold = 0.5):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).int()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        return cm

    def roc_values(self, val_loader):
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu()) 
                all_labels.append(y.cpu())

        all_probs = torch.cat(all_probs).numpy().flatten()
        all_labels = torch.cat(all_labels).numpy().flatten()

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs)

        return fpr, tpr, thresholds, roc_auc
    
    def pr_values(self, val_loader):
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu()) 
                all_labels.append(y.cpu())

        all_probs = torch.cat(all_probs).numpy().flatten()
        all_labels = torch.cat(all_labels).numpy().flatten()

        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)


        return precision, recall, thresholds, pr_auc
