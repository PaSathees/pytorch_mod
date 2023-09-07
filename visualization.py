"""Contains functionality to visualize PyTorch training progress

Currently supports:
    1. Computer Vision

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""
import matplotlib.pyplot as plt

def plot_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """
    curves = results[0]
    loss = curves["train_loss"]
    val_loss = curves["val_loss"]

    accuracy = curves["train_acc"]
    val_accuracy = curves["val_acc"]

    epochs = range(len(curves["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()