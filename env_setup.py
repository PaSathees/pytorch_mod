"""
PyTorch simple function associated with environment setup and device configuration
"""
import torch
import matplotlib
import pandas as pd
import numpy as np

def print_versions():
    """Prints the available packages versions, e.g., PyTorch, Torchinfo...
    """
    # PyTorch
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
    except ImportError:
        print("[INFO] No PyTorch found")

    # Matplotlib
    try:
        import matplotlib
        print(f"Matplotlib Version: {matplotlib.__version__}")
    except ImportError:
        print("[INFO] No Matplotlib found")

    # Pandas
    try:
        import pandas
        print(f"Pandas Version: {pandas.__version__}")
    except ImportError:
        print("[INFO] No Pandas found")

    # Numpy
    try:
        import numpy
        print(f"Numpy Version: {numpy.__version__}")
    except ImportError:
        print("[INFO] No Numpy found")

    # Torchvision
    try:
        import torchvision
        print(f"Torchvision Version: {torchvision.__version__}")
    except ImportError:
        print("[INFO] No Torchvision found")

    # Torchaudio
    try:
        import torchaudio
        print(f"Torchaudio Version: {torchaudio.__version__}")
    except ImportError:
        print("[INFO] No Torchaudio found")

    # Scikit-learn
    try:
        import sklearn
        print(f"Scikit-learn Version: {sklearn.__version__}")
    except ImportError:
        print("[INFO] No Scikit-learn found")

    # Torchmetrics
    try:
        import torchmetrics
        print(f"Torchmetrics Version: {torchmetrics.__version__}")
    except ImportError:
        print("[INFO] No Torchmetrics found")

    # TQDM
    try:
        import tqdm
        print(f"TQDM Version: {tqdm.__version__}")
    except ImportError:
        print("[INFO] No TQDM found")

    # MLXTEND
    try:
        import mlxtend
        print(f"MLEXTEND Version: {mlxtend.__version__}")
    except ImportError:
        print("[INFO] No MLEXTEND found")

    # PIL
    try:
        import PIL
        print(f"PIL Version: {PIL.__version__}")
    except ImportError:
        print("[INFO] No PIL found")

    # Torchinfo
    try:
        import torchinfo
        print(f"Torchinfo Version: {torchinfo.__version__}")
    except ImportError:
        print("[INFO] No Torchinfo found")

    # Gradio
    try:
        import gradio
        print(f"Gradio Version: {gradio.__version__}")
    except ImportError:
        print("[INFO] No Gradio found")


def print_gpu_status():
    """Prints whether a CUDA GPU is available & number of GPUs & prints whether MPS is available
    """
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.device_count()} Supported CUDA GPU available")
    else:
        print("[INFO] No Supported CUDA GPU found")

    if torch.backends.mps.is_available():
        print("[INFO] This device supports MPS")
    else: 
        print("[INFO] This device doesn't support MPS")

def get_agnostic_device():
    """Returns device name as "cuda" if supported GPU is available or "mps" if supported MPS environment is available or will return "cpu"

    Returns:
        string: name of the device ("cuda" or "mps" or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return "mps"
        else:
            print("[WARN] MPS is available, but current version of PyTorch doesn't built with MPS activation. Returning CPU")
            return "cpu"
    else:
        return "cpu"
