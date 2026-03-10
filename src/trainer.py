import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score

import numpy as np
import torch
from src.model import MLP
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

            total_loss += loss.item() * x.size(0)

        return total_loss / len(train_loader.dataset)

    def eval_epoch(self, val_loader):
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.detach().cpu().numpy().reshape(-1))
                all_labels.append(y.detach().cpu().numpy().reshape(-1))

        all_probs = np.concatenate(all_probs)    
        all_labels = np.concatenate(all_labels)  

        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1, best_t = -1.0, 0.5
    
        for t in thresholds:
            preds = (all_probs >= t).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    
        best_preds = (all_probs >= best_t).astype(int)
        acc = accuracy_score(all_labels, best_preds)
        precision = precision_score(all_labels, best_preds, zero_division=0)
        recall = recall_score(all_labels, best_preds, zero_division=0)
        return acc, best_f1, best_t, precision, recall

    def fit(self, train_loader, val_loader, epochs, verbose=False):
        history = {
            "loss": [], 
            "val_acc": [], 
            "val_f1": [], 
            "best_t": [], 
            "val_precision": [], 
            "val_recall": [],
            }

        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            acc, f1, best_t, precision, recall = self.eval_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()


            history["loss"].append(loss)
            history["val_acc"].append(acc)
            history["val_f1"].append(f1)
            history["best_t"].append(best_t)
            history["val_precision"].append(precision)
            history["val_recall"].append(recall)
            if verbose:
                print(f"Epoch {epoch+1} | loss={loss:.4f} | val_acc={acc:.4f} | val_f1={f1:.4f}")

        return history

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path, weights_only=False)

    def confusion_matrix(self, val_loader, threshold = 0.5):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()

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

                all_probs.append(probs.detach().cpu().numpy().reshape(-1))
                all_labels.append(y.detach().cpu().numpy().reshape(-1))

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs)

        return fpr, tpr, thresholds, roc_auc
