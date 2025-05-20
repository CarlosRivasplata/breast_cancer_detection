import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def save_training_history_to_csv(history, path="training_history.csv"):
    """
    Save training history dictionary to CSV.

    Args:
        history (dict): Trainer history from `trainer.get_history()`.
        path (str): File path for CSV.
    """
    df = pd.DataFrame(history)
    df.index.name = "epoch"
    df.to_csv(path)
    print(f"Training history saved to {path}")


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Plot a confusion matrix using seaborn heatmap.

    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels.
        class_names (List[str], optional): Class label names.
        title (str): Plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_history(history, model_name="Model"):
    """
    Plot loss and key evaluation metrics over epochs.

    Args:
        history (dict): Output from `trainer.get_history()`.
        model_name (str): Optional label for the plot title.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(15, 6))

    # --- Plot losses ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss per Epoch")
    plt.legend()

    # --- Plot metrics ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Accuracy")
    plt.plot(epochs, history["val_recall"], label="Recall")
    plt.plot(epochs, history["val_precision"], label="Precision")
    plt.plot(epochs, history["val_f1"], label="F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{model_name} - Metrics per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()
