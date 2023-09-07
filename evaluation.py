"""Contains functionality to predict, and evaluate various PyTorch trained models

Currently supports:
    1. Computer Vision

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""
import torch
import torchvision
from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import requests


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
    

def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
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
            sigmoid_threshold=sigmoid_threshold
        )


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


torch.manual_seed(42)


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


# inference timing evaluation