# PYTORCH BOILERPLATE

Boilerplate modules for PyTorch development and training simplying most repetitive codes targetted for Jupyter notebook environments like Google Colab.

Status:
1. Supports computer vision. Remaining tasks:
- [ ] Prediction function
- [ ] Evaluation function
- [ ] Chart visualization function
- [x] Update tensorboard with train_engine.py
- [ ] Testing with FoodVision Mini
- [ ] Testing with FoodVision Big
- [ ] Import modules section

Import module to Google Colab By:
`

`
Install module requirements by:
1. Manually installing PyTorch: 
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
or visit https://pytorch.org
2. Install rest of the requirements: `pip install -r requirements.txt`

Includes following modules:
1. [env_setup.py](env_setup.py) : Includes following functions related setting up the environment and device agnostic code with PyTorch: 
   - `print_versions()`: Prints the available packages versions, e.g., PyTorch, Torchinfo...
   - `print_gpu_status()`: Prints whether a CUDA GPU is available & number of GPUs
   - `get_agnostic_device()`: Returns device name as "cuda" if supported GPU is available or will return "cpu"
2. [data_setup.py](data_setup.py) : Includes following functions to setting up data for training:
   - `create_cv_dataloaders(train_dir, test_dir, transform, batch_size, val_dir, num_workers)`: Creates training, validation (optional: if `val_dir` is provided), and testing DataLoaders
3. [engine.py](engine.py) : Inlcudes following functions related to trianing a PyTorch model in a device agnostic manner:
   - `train_step(model, dataloader, loss_fn, optimizer, device)`: Training loop for a single epoch with PyTorch
   - `test_step(model, dataloader, loss_fn, device)`: Testing loop for single epoch with PyTorch
   - `train(model, train_dataloader, optimizer, loss_fn, epochs, device, val_dataloader, test_dataloader, print_status)`: Trains, validates (optional), and tests a PyTorch Model
4. [evaluation.py](evaluation.py) : Contains following functions to predict, and evaluate various PyTorch trained models: 
5. [deployment.py](deployment.py) : Contains following functions for deploying PyTorch models:
   - `save_model_to_directory(model:, target_directory, save_name)`: Saves PyTorch model to a local target directory
6. [cv_models.py](cv_models.py) : Contains following of the state-of-the-art PyTorch computer vision model architectures:
   - `TinyVGG(torch.nn.Module)`: Creates the TinyVGG architecture: https://poloclub.github.io/cnn-explainer/
7. [visualization.py](visualization.py) : Contains following functions to visualize metrics:
8. [experimentation.py](experimentation.py) : Contains following functions for experimenting with PyTorch:
   - `create_writer(experiment_name, model_name, extra)` : Creates a torch.utils.tensorboard.writer.SummaryWriter() to a specific log_dir