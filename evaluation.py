"""Contains functionality to predict, and evaluate various PyTorch trained models

Currently supports:
    1. Computer Vision

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""
import random
from typing import List
from pathlib import Path
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import requests
from google.colab import files


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    device: torch.device,
    transform: torchvision.transforms,
    multiclass: bool = True,
    sigmoid_threshold: float = 0.5,
):
    """Predicts a local image with the given model and plots both predictions and image

    Args:
        model (torch.nn.Module): Model to predict on,
        image_path (str): Path to local image,
        class_names (List[str]): List of class names,
        device (torch.device): Device for inference,
        transform (torchvision.transforms): Transforms for image,
        multiclass (bool, optional): whether prediction is for multiclass, Default True.
        sigmoid_threshold (float, optional): if prediction is binary, sigmoid threshold value. Default 0.5
    """

    img = Image.open(image_path)

    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    if multiclass:
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    else:
        target_image_pred_label = 1 if target_image_pred > sigmoid_threshold else 0

    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)


def pred_on_custom_image_url(
    model: torch.nn.Module,
    class_names: List[str],
    device: torch.device,
    transform: torchvision.transforms,
    url: str,
    custom_image_path: str = "test.jpeg",
    multiclass: bool = True,
    sigmoid_threshold: float = 0.5,
):
    """Predicts on a custom image by downloading from the given URL with the given model and plots both predictions and image

    Args:
        model (torch.nn.Module): Model to predict on,
        class_names (List[str]): List of class names,
        device (torch.device): Device for inference,
        transform (torchvision.transforms): Transforms for image,
        url (str): URL of the image,
        custom_image_path (str, optional): Path to local image, Default "test.jpg",
        multiclass (bool, optional): whether prediction is for multiclass, Default True.
        sigmoid_threshold (float, optional): if prediction is binary, sigmoid threshold value. Default 0.5
    """
    with open(custom_image_path, "wb") as file:
        request = requests.get(url)
        print(f"Downloading {custom_image_path}...")
        file.write(request.content)

        pred_and_plot_image(
            model=model,
            image_path=custom_image_path,
            class_names=class_names,
            device=device,
            transform=transform,
            multiclass=multiclass,
            sigmoid_threshold=sigmoid_threshold,
        )


def pred_and_plot_local_random_images(
    model: torch.nn.Module,
    test_dir_path: str,
    class_names: List[str],
    device: torch.device,
    transform: torchvision.transforms,
    multiclass: bool = True,
    image_extension: str = ".jpg",
    sigmoid_threshold: float = 0.5,
    num_images_to_plot: int = 3,
):
    """Predicts random number of local image with the given model and plots both predictions and images

    Args:
        model (torch.nn.Module): Model to predict on,
        test_dir_path (str): Path to local dir that contains test images,
        class_names (List[str]): List of class names,
        device (torch.device): Device for inference,
        transform (torchvision.transforms): Transforms for image,
        multiclass (bool, optional): whether prediction is for multiclass, Default True.
        image_extension (bool): Extension of image files. Default ".jpg"
        sigmoid_threshold (float, optional): if prediction is binary, sigmoid threshold value. Default 0.5
        num_images_to_plot (int): Number of images to plot, default 3.
    """

    # Select random number of image paths from test_dir
    test_image_path_list = list(Path(test_dir_path).glob(f"*/*{image_extension}"))
    test_image_path_sample = random.sample(
        population=test_image_path_list, k=num_images_to_plot
    )

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        pred_and_plot_image(
            model=model,
            image_path=image_path,
            class_names=class_names,
            device=device,
            transform=transform,
            multiclass=multiclass,
            sigmoid_threshold=sigmoid_threshold,
        )


def pred_and_plot_colab_interface(
    model: torch.nn.Module,
    class_names: List[str],
    device: torch.device,
    transform: torchvision.transforms,
    multiclass: bool = True,
    sigmoid_threshold: float = 0.5,
):
    """Predicts images uploaded with Google Colab upload interface with the given model and plots both predictions and images

    IMPORTANT; Will not work in environments other than Google Colab

    Args:
        model (torch.nn.Module): Model to predict on,
        test_dir_path (str): Path to local dir that contains test images,
        class_names (List[str]): List of class names,
        device (torch.device): Device for inference,
        transform (torchvision.transforms): Transforms for image,
        multiclass (bool, optional): whether prediction is for multiclass, Default True.
        image_extension (bool): Extension of image files. Default ".jpg"
        sigmoid_threshold (float, optional): if prediction is binary, sigmoid threshold value. Default 0.5
        num_images_to_plot (int): Number of images to plot, default 3.
    """
    # Upload files using Google Colab interface
    uploaded = files.upload()

    for file_name in uploaded.keys():
        # predicting images
        image_path = "/content/" + file_name

        pred_and_plot_image(
            model=model,
            image_path=image_path,
            class_names=class_names,
            device=device,
            transform=transform,
            multiclass=multiclass,
            sigmoid_threshold=sigmoid_threshold,
        )


def eval_model(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
):
    """Returns a dictionary containing the results of model predicting on test_dataloader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on test_dataloader.
        test_dataloader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on test_dataloader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(
                y_true=y, y_pred=y_pred.argmax(dim=1)
            )  # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        loss /= len(test_dataloader)
        acc /= len(test_dataloader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc,
    }
