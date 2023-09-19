"""Contains functionality for preparing data to PyTorch DataLoader.

Currently supports:
    1. Computer Vision

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

NUM_WORKERS = os.cpu_count()


def create_cv_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
    val_dir: str = None,
    val_transform: transforms.Compose = None,
    num_workers: int = NUM_WORKERS,
):
    """Creates training, validation (optional), and testing DataLoaders.

    Takes training, validation (optional), and testing directory paths and returns respective PyTorch DataLoaders.

    Args:
        train_dir (str): Training directory path
        test_dir (str): Testing directory path
        train_transform (transforms.Compose): torchvision transforms for training data
        test_transform (transforms.Compose): torchvision transforms for testing data
        batch_size (int): number of samples per batch
        device (torch.device): PyTorch device for Dataloader Generator
        val_dir (str, optional): Validation directory path. Defaults to None.
        val_transform (transforms.Compose): torchvision transforms for validation data
        num_workers (int, optional): Number of workers per DataLoader. Defaults to NUM_WORKERS.

    Returns:
        Tuple: (train_dataloader, test_dataloader, class_names) or (train_dataloader, val_dataloader, test_dataloader, class_names)
    """
    # Creating datasets from Image folder paths
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator(device)
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator(device)
    )

    if val_dir:
        assert (
            val_transform
        ), "[WARN] [EXIT] val_transform should be provided, for validation"
        val_data = datasets.ImageFolder(val_dir, transform=val_transform)
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=torch.Generator(device)
        )

        return train_dataloader, val_dataloader, test_dataloader, class_names

    return train_dataloader, test_dataloader, class_names
