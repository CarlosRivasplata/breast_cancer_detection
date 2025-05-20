# training/engine.py

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm.notebook import tqdm

from training.early_stopping import EarlyStopping


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Run one epoch of training.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate_epoch(model, loader, criterion, device):
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Validation/test data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.

    Returns:
        Tuple[float, float, float]: (accuracy %, avg loss, macro recall)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return accuracy, avg_loss, recall, precision, f1


class Trainer:
    """
    Encapsulates the training and evaluation logic for a PyTorch model like ResNet, EfficientNet, etc.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 optimizer: Optimizer, criterion: nn.Module,
                 scheduler: ReduceLROnPlateau = None,
                 early_stopping: EarlyStopping = None):
        """
        Initialize Trainer.

        Args:
            model (nn.Module): Model to train.
            device (torch.device): CUDA or CPU device.
            optimizer (Optimizer): Optimizer to use (e.g., Adam).
            criterion (nn.Module): Loss function.
            scheduler (ReduceLROnPlateau, optional): Learning rate scheduler.
            early_stopping: EarlyStopping class to stop before overfitting.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_recall": [],
            "val_precision": [],
            "val_f1": [],
        }

        self.best_model_wts = None
        self.best_val_recall = float("-inf")
        self.save_path = None
        self.early_stopping = early_stopping


    def fit(self, train_loader, val_loader, epochs=20):
        """
        Train the model for a given number of epochs.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = train_epoch(self.model, train_loader, self.criterion, self.optimizer, self.device)
            val_acc, val_loss, val_recall, val_precision, val_f1 = evaluate_epoch(
                self.model, val_loader, self.criterion, self.device
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc) # Normalizing this one
            self.history["val_recall"].append(val_recall)
            self.history["val_precision"].append(val_precision)
            self.history["val_f1"].append(val_f1)

            if self.scheduler:
                self.scheduler.step(val_acc)

            # Track the best model by validation recall
            # I want to penalize the false negatives the most
            if val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
                self.best_model_wts = self.model.state_dict()

            print(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | Recall: {val_recall:.4f} | "
                f"Precision: {val_precision:.4f} | F1: {val_f1:.4f}"
            )

            if self.early_stopping:
                # Value lookup
                monitored_value = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_recall": val_recall,
                    "val_precision": val_precision,
                    "val_f1": val_f1,
                }.get(self.early_stopping.monitor)

                if monitored_value is None:
                    raise ValueError(f"Unknown metric: {self.early_stopping.monitor}. Valid metrics are {list(monitored_value.keys())}")

                stop = self.early_stopping.step(monitored_value)
                if stop:
                    print(f"Early stopping triggered at epoch {epoch + 1} (no improvement in {self.early_stopping.monitor}).")
                    break

    def evaluate(self, test_loader):
        """
        Evaluate the trained model on a test set.

        Args:
            test_loader (DataLoader): Test data.

        Returns:
            Tuple[float, float, float, float, float]:
            (accuracy %, avg loss, recall, precision, f1)
        """
        return evaluate_epoch(self.model, test_loader, self.criterion, self.device)

    def get_history(self):
        """
        Get training history.

        Returns:
            dict: Training metrics over all epochs.
        """
        return self.history

    def get_model(self):
        """
        Get the trained model.

        Returns:
            nn.Module: Trained PyTorch model.
        """
        return self.model

    def save_best_model(self, save_path: str = "models/best_model.pth"):
        """
        Save the best model weights (based on validation accuracy).

        Args:
            save_path (str): File path to save the model weights.
        """
        self.save_path = save_path
        if self.best_model_wts:
            torch.save(self.best_model_wts, self.save_path)
            print(f"Best model saved to: {self.save_path} (val_recall: {self.best_val_recall:.2f}%)")
        else:
            print("No best model weights available to save.")

    def load_model(self, path: str):
        """
        Load model weights from a file.

        Args:
            path (str): File path of the saved weights.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"Model loaded from: {path}")
