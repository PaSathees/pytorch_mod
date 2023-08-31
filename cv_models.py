"""Creates a PyTorch computer vision model architectures

Currently supports:
    1. TinyVGG

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""

import torch


class TinyVGG(torch.nn.Module):
    """Creates the TinyVGG architecture: https://poloclub.github.io/cnn-explainer/

    Args:
        input_shape (int): Number of input channels
        hidden_units (int): Number units of hidden layers
        output_shape (int): Number of output channels
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=hidden_units
                * 13
                * 13,  # set incorrect & find with error message
                out_features=output_shape,
            ),
        )

    def forward(self, x: torch.Tensor):
        """Implements forward pass of TinyVGG

        Args:
            x (torch.Tensor): Training tensor

        Returns:
            classification (torch.tensor): Forward pass results
        """
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
