import torch
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

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
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        return acc

    def fit(self, train_loader, val_loader, epochs, verbose=False):
        history = {"loss": [], "val_acc": []}

        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            acc = self.eval_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            history["loss"].append(loss)
            history["val_acc"].append(acc)

            if verbose:
                print(f"Epoch {epoch+1} | loss={loss:.4f} | val_acc={acc:.4f}")

        return history
