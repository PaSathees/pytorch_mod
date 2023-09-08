"""Contains functionality to visualize PyTorch training progress

Currently supports:
    1. Computer Vision

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision


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


def plot_random_images(
    dir_path: str, file_pattern: str = "*/*/*.jpg", num_plots: int = 3, seed: int = 42
):
    """Plots random images from the given image directory.

    Args:
        dir_path (str): Image path of data image directory
        file_pattern (str): Pattern and extension of the files. Default "*/*/*.jpg".
        num_plots (int): Number of plots to plot, Default 3.
        seed (int): Random seed, Default 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get image paths
    image_paths = list(dir_path.glob(file_pattern))
    random_image_paths = random.sample(image_paths, num_plots)

    for random_img_path in random_image_paths:
        # Get random images
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


def plot_random_transformed_images(
    dir_path: str,
    transform: torchvision.transforms,
    file_pattern: str = "*/*/*.jpg",
    num_plots: int = 3,
    seed: int = 42,
):
    """
    Selects random images from a path of images and loads/transforms
    them then plots the original vs the transformed version.

    Args:
        dir_path (str): Image path of data image directory
        transform (torchvision.transforms): Transforms to test
        file_pattern (str): Pattern and extension of the files. Default "*/*/*.jpg".
        num_plots (int): Number of plots to plot, Default 3.
        seed (int): Random seed, Default 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get image paths
    image_paths = list(dir_path.glob(file_pattern))
    random_image_paths = random.sample(image_paths, num_plots)

    for random_img_path in random_image_paths:
        with Image.open(random_img_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            # Transform and plot target image
            transformed_image = transform(f).permute(
                1, 2, 0
            )  # Change shape for matplotlib (C, H, W) -> (H, W, C)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")
            fig.suptitle(f"Class: {random_img_path.parent.stem}", fontsize=16)
