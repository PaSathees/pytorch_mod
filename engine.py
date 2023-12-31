"""Contians training functions for PyTorch model

Currently supports:
    1. Computer Vision (multiclass and binary)

Authored by:
    1. Sathees Paskaran

Modified by:

License: MIT
"""

from typing import Dict, List, Tuple
from timeit import default_timer as timer
import torch
from tqdm.auto import tqdm
from torch.utils import tensorboard
import torchmetrics


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    problem_type: str = "multiclass",
    sigmoid_threshold: float = 0.5
) -> Tuple[float, float]:
    """Training loop for a single epoch with PyTorch

    Trains given PyTorch model through necessary trianing steps

    Args:
        model (torch.nn.Module): PyTorch Model
        dataloader (torch.utils.data.DataLoader): DataLoader instance for training
        loss_fn (torch.nn.Module): PyTorch loss function to minimize
        optimizer (torch.optim.Optimizer): PyTorch optimizer to help minimize loss function
        device (torch.device): PyTorch device instance
        problem_type (str): Type of problem (values: "multiclass", "binary"). Default "multiclass"
        sigmoid_threshold (float): Sigmoid threshold value, default 0.5

    Returns:
        Tuple (float, float): results of epoch training (loss, accuracy)
    """
    # Set trianing mode
    model.train()

    # define train loss & acc variables
    loss = 0
    all_predictions = torch.Tensor([]).to(device)
    all_targets = torch.Tensor([]).to(device)

    # loop through batches
    for batch, (X, y) in enumerate(dataloader):
        #add additional dimension for binary classification
        if problem_type == "binary":
            y = y.unsqueeze(1).long()
            y = y.float()

        # send data to device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        batch_loss = loss_fn(y_pred, y)
        loss += batch_loss.item()

        # 3. Optimer set zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        batch_loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Accumulate for accuracy
        if problem_type == "multiclass":
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        elif problem_type == "binary":
            y_pred_class = (y_pred >= sigmoid_threshold).float()

        all_predictions = torch.cat((all_predictions, y_pred_class), dim=0)
        all_targets = torch.cat((all_targets, y), dim=0)

    # calculate train loss & accuracy for epoch
    loss = loss / len(dataloader)
    accuracy = torchmetrics.Accuracy(
        task=problem_type, 
        num_classes=len(dataloader.dataset.classes)).to(device)
    acc = accuracy(all_predictions, all_targets).item()

    return loss, acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    problem_type: str = "multiclass",
    sigmoid_threshold: float = 0.5
) -> Tuple[float, float]:
    """Testing loop for single epoch with PyTorch

    Performs a epoch testing loop on a PyTorch Model with "eval" mode.

    Args:
        model (torch.nn.Module): PyTorch model to be tested
        dataloader (torch.utils.data.DataLoader): DataLoader instance to be tested on
        loss_fn (torch.nn.Module): PyTorch loss function to test
        device (torch.device): PyTorch device instance for testing
        problem_type (str): Type of problem (values: "multiclass", "binary"). Default "multiclass"
        sigmoid_threshold (float): Sigmoid threshold value, default 0.5

    Returns:
        Tuple (float, float): Testing results (loss, accuarcy)
    """
    # set eval mode
    model.eval()

    # define loss & accuracy values
    loss = 0
    all_predictions = torch.Tensor([]).to(device)
    all_targets = torch.Tensor([]).to(device)

    # Set inference mode
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            #add additional dimension for binary classification
            if problem_type == "binary":
                y = y.unsqueeze(1)
                y = y.float()

            # Send data to device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss
            batch_loss = loss_fn(y_pred, y)
            loss += batch_loss.item()

            # Accumulate for accuracy
            if problem_type == "multiclass":
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            elif problem_type == "binary":
                y_pred_class = (y_pred >= sigmoid_threshold).float()

            all_predictions = torch.cat((all_predictions, y_pred_class), dim=0)
            all_targets = torch.cat((all_targets, y), dim=0)

    # Calculate test loss and accuracy for epoch
    loss = loss / len(dataloader)
    accuracy = torchmetrics.Accuracy(
        task=problem_type, 
        num_classes=len(dataloader.dataset.classes)).to(device)
    acc = accuracy(all_predictions, all_targets).item()

    return loss, acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    val_dataloader: torch.utils.data.DataLoader = None,
    test_dataloader: torch.utils.data.DataLoader = None,
    print_status: bool = True,
    writer: tensorboard.SummaryWriter = None,
    problem_type: str = "multiclass",
    sigmoid_threshold: float = 0.5
) -> Dict[str, List[float]]:
    """Trains, validates (optional), and tests a PyTorch Model.

    Important: If you only have train & test dataloaders provide your test dataloader as val_dataloader and don't set test_dataloder. If you have train, validation & test dataloaders set all three. Like this,
        if train_dataloader & test_dataloader:
            train_dataloader = train_dataloader
            val_dataloader = test_dataloader
            test_dataloader = None
        if train_dataloader, val_dataloader, & test_dataloader:
            train_dataloader = train_dataloader
            val_dataloader = val_dataloader
            test_dataloader = test_dataloader

    Trains the PyTorch model for given number of epochs by passing through train_step() and test_step().

    Args:
        model (torch.nn.Module): PyTorch model to be trained and tested
        train_dataloader (torch.utils.data.DataLoader): Training DataLoader instance
        optimizer (torch.optim.Optimizer): PyTorch Optimizer to help minimize loss function
        loss_fn (torch.nn.Module): PyTorch loss function to calculate loss
        epochs (int): Number of epochs
        device (torch.device): PyTorch device instance
        test_dataloader (torch.utils.data.DataLoader, optional): DataLoader instance to test the model. Defaults to None.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader instance to validate the model. Defaults to None.
        print_status (bool, optional): Whether to print epoch results. Defaults to True.
        writer (torch.utils.tensorboard.writer.SummaryWriter, optional): A SummaryWriter instance to write model summary to.
        problem_type (str): Type of problem (values: "multiclass", "binary"). Default "multiclass"
        sigmoid_threshold (float): Sigmoid threshold value, default 0.5

    Returns:
        Tuple (Dict[str, List[float]], float): Results dictionary with training and testing loss and accuracies over the epochs, and total training time
        In the form: ({
            train_loss: [...],
            train_acc: [...],
            val_loss: [...],
            val_acc: [...]
            },
            67.5)
    """

    # Device check
    print(f"[INFO] Using device: {device}")

    # Parameter check
    assert (
        (val_dataloader)
    ), "[WARN] [EXIT] val_dataloader should be provided, if you have only train & test, set val_dataloader = test_dataloader & ignore test_dataloader parameter"

    # Define results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Starting timer
    start_time = timer()

    # Training through epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            problem_type=problem_type,
            sigmoid_threshold=sigmoid_threshold
        )

        val_loss, val_accuracy = test_step(
            model=model, 
            dataloader=val_dataloader, 
            loss_fn=loss_fn, 
            device=device, 
            problem_type=problem_type,
            sigmoid_threshold=sigmoid_threshold
        )

        # Printing status
        if print_status:
            print(
                f"[INFO] Epoch: {epoch+1} | "
                f"Train_loss: {train_loss:.4f} | "
                f"Train_acc: {train_accuracy:.4f} | "
                f"Val_loss: {val_loss:.4f} | "
                f"Val_acc: {val_accuracy:.4f}"
            )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_accuracy)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_accuracy)

        # Use writer if given
        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "val_loss": val_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_accuracy, "val_acc": val_accuracy},
                global_step=epoch,
            )

            if not test_dataloader:
                writer.close()

    # print timing
    training_time = timer() - start_time
    print(f"[INFO] Training time: {training_time:.3f} seconds")

    if test_dataloader:
        test_loss, test_accuracy = test_step(
            model=model, 
            dataloader=test_dataloader, 
            loss_fn=loss_fn, 
            device=device, 
            problem_type=problem_type,
            sigmoid_threshold=sigmoid_threshold
        )

        # printing status
        if print_status:
            print(
                f"[INFO] test_loss: {test_loss:.4f} | " f"test_acc: {test_accuracy:.4f}"
            )

        results["test_loss"] = test_loss
        results["test_accuracy"] = test_accuracy

        if writer:
            writer.add_scalars(
                main_tag="Test",
                tag_scalar_dict={"test_loss": test_loss, "test_acc": test_accuracy},
                global_step=epochs - 1,
            )

            writer.close()

    return results, training_time