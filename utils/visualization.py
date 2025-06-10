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
    base_history = {k: v for k, v in history.items() if k != "val_views"}
    df = pd.DataFrame(base_history)
    df.index.name = "epoch"
    df.to_csv(path)
    print(f"Training history (global metrics) saved to {path}")

    # Save per-view metrics separately
    if "val_views" in history:
        view_records = []
        for (view, metric), values in history["val_views"].items():
            for epoch, value in enumerate(values):
                view_records.append({
                    "epoch": epoch,
                    "view": view,
                    "metric": metric,
                    "value": value
                })
        view_df = pd.DataFrame(view_records)
        view_path = path.replace(".csv", "_views.csv")
        view_df.to_csv(view_path, index=False)
        print(f"Per-view metrics saved to {view_path}")



def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", cmap="Blues"):
    """
    Plot a confusion matrix using seaborn heatmap.

    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels.
        class_names (List[str], optional): Class label names.
        title (str): Plot title.
        cmap (str): Matplotlib colormap name for the heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    

def plot_training_history(history, model_name="Model"):
    """
    Plot loss and key evaluation metrics over epochs with vertical lines at each epoch.

    Args:
        history (dict): Output from `trainer.get_history()`.
        model_name (str): Optional label for the plot title.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    for epoch in epochs:
        plt.axvline(x=epoch, color="gray", linestyle="--", linewidth=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss per Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Accuracy")
    plt.plot(epochs, history["val_recall"], label="Recall")
    plt.plot(epochs, history["val_precision"], label="Precision")
    plt.plot(epochs, history["val_f1"], label="F1 Score")
    plt.plot(epochs, history["val_auc"], label="AUC-ROC")
    for epoch in epochs:
        plt.axvline(x=epoch, color="gray", linestyle="--", linewidth=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{model_name} - Metrics per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()

    if "val_views" in history:
        plt.figure(figsize=(10, 5))
        for (view, metric), values in history["val_views"].items():
            if metric == "recall":
                plt.plot(epochs[:len(values)], values, label=f"{view} recall")
        plt.title(f"{model_name} - Per-View Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if "epoch_time" in history and "max_memory_mb" in history:
        plot_training_profiling(history, model_name)

def plot_training_profiling(history, model_name="Model"):
    """
    Plot training time and peak memory usage per epoch.

    Args:
        history (dict): Output from `trainer.get_history()`.
        model_name (str): Optional label for the plot title.
    """
    epochs = range(1, len(history["epoch_time"]) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot training time
    ax1.plot(epochs, history["epoch_time"], label="Epoch Time (s)", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Time (s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(f"{model_name} - Profiling Metrics")

    # Add memory on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["max_memory_mb"], label="Max Memory (MB)", color="tab:red")
    ax2.set_ylabel("Memory (MB)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    fig.legend(loc="upper right")
    plt.show()