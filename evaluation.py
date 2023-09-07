"""Contains functionality to predict, and evaluate various PyTorch trained models

Currently supports:
    1. Computer Vision (multiclass and binary)

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""
import random
from typing import List, Tuple
from pathlib import Path
import torch
from PIL import Image
import torchmetrics
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import requests
from google.colab import files
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from mlxtend.plotting import plot_confusion_matrix
import torchvision.transforms as transforms


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


def evaluate_model_metrics(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List,
    task: str,
    loss_fn: torch.nn.Module,
    average: str = "micro",
    threshold: float = 0.5,
):
    """
    Evaluate a PyTorch model using torchmetrics for common metrics.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        test_dataloader (DataLoader): DataLoader containing the test data.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').
        class_names (List): List of class names.
        task (str): Task name (e.g., 'binary', 'multiclass' or 'multilabel').
        loss_fn (torch.nn.Module): loss function for loss evaluation
        average (str): Torchmetrics average parameter, Default "micro"
        threshold (float): Sigmoid threshold, Default 0.5.

    Returns:
        dict: Dictionary containing various evaluation metrics.
    """
    metric_dict = {}
    num_classes = len(class_names)

    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(device)
    precision = torchmetrics.Precision(
        task=task, average=average, num_classes=num_classes
    ).to(device)
    recall = torchmetrics.Recall(
        task=task, average=average, num_classes=num_classes
    ).to(device)
    f1_score = torchmetrics.F1Score(task=task, num_classes=num_classes).to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(
        task=task, num_classes=num_classes
    ).to(device)

    # Metric calculation
    loss = 0
    outputs = []
    targets = []
    model.eval()
    with torch.inference_mode():
        # Loss calculation & accumulating outputs and targets for other metrics
        for X, y in tqdm(test_dataloader, desc="Making Predictions"):
            y_pred_logits = model(X)
            loss += loss_fn(y_pred_logits, y)

            if task == "multiclass":
                y_pred_probs = torch.softmax(y_pred_logits, dim=1)
                y_pred_label = torch.argmax(y_pred_probs, dim=1)
            else:
                y_pred_label = 1 if y_pred_logits > threshold else 0

            outputs.append(y_pred_label.cpu())
            targets.append(y.cpu())

        loss /= len(test_dataloader)
        outputs_tensor = torch.cat(outputs)
        targets_tensor = torch.cat(targets)

        # Other metrics
        metric_dict = {
            "model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": accuracy(outputs_tensor, targets_tensor).item(),
            "model_precision": precision(outputs_tensor, targets_tensor).item(),
            "model_recall": recall(outputs_tensor, targets_tensor).item(),
            "model_fl_score": f1_score(outputs_tensor, targets_tensor).item(),
            "model_confusion_matrix": confusion_matrix(outputs_tensor, targets_tensor)
            .cpu()
            .numpy(),
        }

    # Plot the confusion matrix
    plot_confusion_matrix(
        conf_mat=metric_dict["model_confusion_matrix"],
        class_names=class_names,
        figsize=(10, 7),
    )

    return metric_dict


def evaluate_classification_report(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List,
    task: str,
    threshold: float = 0.5,
):
    """
    Evaluate a PyTorch model and generate a classification report.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        test_dataloader (DataLoader): DataLoader containing the test data.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').
        class_names (List): List of class names.
        task (str): Task name (e.g., 'binary', 'multiclass' or 'multilabel').
        threshold (float): Sigmoid threshold, Default 0.5.

    Returns:
        str: Classification report as a string.
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.inference_mode():
        for inputs, targets in tqdm(test_dataloader, desc="Making Predictions"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if task == "multiclass":
                y_pred_probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(y_pred_probs, dim=1)
            else:
                predicted = 1 if outputs > threshold else 0
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=class_names)
    return report


def failed_image_generator(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    task: str,
    threshold: float = 0.5,
):
    """
    Generator that yields failed images from the test DataLoader with their predicted and target labels.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        test_dataloader (DataLoader): DataLoader containing the test data.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').
        task (str): Task name (e.g., 'binary', 'multiclass' or 'multilabel').
        threshold (float): Sigmoid threshold, Default 0.5.

    Returns:
        generator: Returns generator of (input, prediction, target)
    """
    model.eval()

    with torch.inference_mode():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if task == "multiclass":
                y_pred_probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(y_pred_probs, dim=1)
            else:
                predictions = 1 if outputs > threshold else 0
            for input, prediction, target in zip(inputs, predictions, targets):
                if not torch.all(prediction == target):
                    yield input.cpu(), prediction.cpu(), target.cpu()


def inverse_normalize(tensor: torch.Tensor, mean: Tuple, std: Tuple):
    """
    Inverse normalization of a PyTorch tensor.

    Args:
        tensor (torch.Tensor): The tensor to be normalized.
        mean (tuple): Mean values used for normalization.
        std (tuple): Standard deviation values used for normalization.

    Returns:
        torch.Tensor: The de-normalized tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def plot_failed_images_from_generator(
    failed_image_gen,
    class_names: List,
    transform: torchvision.transforms = None,
    max_images: int = 10,
):
    """
    Plot failed images from the generator with their predicted and target labels.

    Args:
        failed_image_gen: Generator that yields failed images.
        class_names (List): List of class names.
        transform (torchvision.transforms): Transform used for normalization.
        max_images (int): Maximum number of failed images to plot (default is 10).
    """
    for i, (image, predicted, target) in enumerate(failed_image_gen):
        plt.figure(figsize=(8, 4))

        if transform:
            # Inverse normalize the image tensor
            mean = transform.mean
            std = transform.std
            denormalized_image = inverse_normalize(image.clone(), mean, std)
        else:
            denormalized_image = image.clone()

        # Convert the tensor to a PIL image
        denormalized_image = transforms.ToPILImage()(denormalized_image)

        # Plot incorrect images
        plt.imshow(denormalized_image)
        plt.title(
            f"Predicted: {class_names[predicted.item()]} | Ground Truth: {class_names[target.item()]}"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        if i + 1 >= max_images:
            break
