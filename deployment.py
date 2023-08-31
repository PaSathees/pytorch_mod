"""Contains functionality for deploying PyTorch models.

Currently supports:
    1. Computer Vision

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""

from pathlib import Path
import torch


def save_model_to_directory(
    model: torch.nn.Module, target_directory: str, save_name: str
) -> str:
    """Saves PyTorch model to a local target directory

    Args:
        model (torch.nn.Module): PyTorch model to save
        target_directory (str): Local directory to save to
        save_name (str): Model's filename. Should end in ".pth" or ".pt"

    Returns:
        str: Path of saved model
    """
    # Create directory structure
    directory_path = Path(target_directory)
    directory_path.mkdir(parents=True, exist_ok=True)

    # Create save path
    assert save_name.endswith(".pth") or save_name.endswith(
        "pt"
    ), "[WARN] [EXIT] PyTorch model name should end with '.pth' or 'pt'"
    save_path = target_directory / save_name

    # Save model's state_dict() to the save path
    torch.save(obj=model.state_dict(), f=save_path)
    print(f"[INFO] Model saved to: {save_path}")
    return save_path
