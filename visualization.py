"""Contains functionality to visualize PyTorch training progress

Currently supports:
    1. Computer Vision

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np


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


NUM_PLOTS = 2


def plot_random_images(
    dir_path: str, file_pattern: str = "*/*/*.jpg", num_plots: int = 3
):
    """Plots random images from the given image directory.

    Args:
        dir_path (str): Image path of data image directory
        file_pattern (str): Pattern and extension of the files. Default "*/*/*.jpg".
        num_plots (int): Number of plots to plot, Default 3.
    """
    # Get image paths
    image_paths = list(dir_path.glob(file_pattern))

    for _ in range(num_plots):
        # Get random images
        random_img_path = random.choice(image_paths)
        image_class = random_img_path.parent.stem
        img = Image.open(random_img_path)

        # Turn the image into an array
        img_array = np.asarray(img)

        # Plot the image with matplotlib
        plt.figure(figsize=(10, 7))
        plt.imshow(img_array)
        plt.title(
            f"Image class: {image_class} | Image shape: {img_array.shape} -> (HWC)"
        )
        plt.axis(False)
