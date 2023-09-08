# PYTORCH BOILERPLATE

Boilerplate modules for PyTorch development and training simplying most repetitive codes targetted for Jupyter notebook environments like Google Colab.

Status:
1. Supports computer vision. Remaining tasks:
- [ ] Prediction function
- [ ] Evaluation function
- [ ] Testing with FoodVision Mini
- [ ] Testing with FoodVision Big

Import module to Google Colab By:
<br>`!rm -rf pytorch_mod`
<br>`!git clone https://github.com/PaSathees/pytorch_mod.git`

Sample import: 
<br>`from pytorch_mod import env_setup, data_setup, engine, evaluation, utils, visualization, experimentation, deployment, cv_model_builders`

Install module requirements by:
1. Manually installing PyTorch: 
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
or visit https://pytorch.org
2. Install rest of the requirements: `pip install -r pytorch_mod/requirements.txt --quiet`

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
   - `train(model, train_dataloader, optimizer, loss_fn, epochs, device, val_dataloader, test_dataloader, print_status)`:Trains, validates (optional), and tests a PyTorch Model
4. [evaluation.py](evaluation.py) : Contains following functions to predict, and evaluate various PyTorch trained models: 
   - `pred_and_plot_image(model, image_path, class_names, device, ransform, multiclass, sigmoid_threshold)`: Predicts a local image with the given model and plots both predictions and image
   - `pred_on_custom_image_url(model, class_names, device, transform, url, custom_image_path, multiclass, sigmoid_threshold)`: Predicts on a custom image by downloading from the given URL with the given model and plots both predictions and image
   - `pred_and_plot_local_random_images(model, test_dir_path, class_names, device, transform, multiclass, image_extension, sigmoid_threshold, num_images_to_plot)`: Predicts random number of local image with the given model and plots both predictions and images
   - `pred_and_plot_colab_interface(model, class_names, device, transform, multiclass, sigmoid_threshold)`: Predicts images uploaded with Google Colab upload interface with the given model and plots both predictions and images
   - `evaluate_model_metrics(model, test_dataloader, device, class_names, task, loss_fn, average, threshold)`: Evaluate a PyTorch model using torchmetrics for common metrics.
   - `evaluate_classification_report(model, test_dataloader, device, class_names, task, threshold)`: Evaluate a PyTorch model and generate a classification report.
   - `failed_image_generator(model, test_dataloader, device, task, threshold)`: Generator that yields failed images from the test DataLoader with their predicted and target labels.
   - `inverse_normalize(tensor, mean, std)`: Inverse normalization of a PyTorch tensor.
   - `plot_failed_images_from_generator(failed_image_gen, class_names, transform, max_images)`: Plot failed images from the generator with their predicted and target labels.
5. [deployment.py](deployment.py) : Contains following functions for deploying PyTorch models:
   - `save_model_to_directory(model:, target_directory, save_name)`: Saves PyTorch model to a local target directory
6. [cv_model_builders.py](cv_model_builders.py) : Contains following of the state-of-the-art PyTorch computer vision model architectures:
   - `TinyVGG(torch.nn.Module)`: Creates the TinyVGG architecture: https://poloclub.github.io/cnn-explainer/
7. [visualization.py](visualization.py) : Contains following functions to visualize metrics:
   - `plot_curves(results)`: Plots training curves of a training results dictionary.
   - `plot_random_images(dir_path, file_pattern, num_plots)`: Plots random images from the given image directory.
8. [experimentation.py](experimentation.py) : Contains following functions for experimenting with PyTorch:
   - `create_writer(experiment_name, model_name, extra)` : Creates a torch.utils.tensorboard.writer.SummaryWriter() to a specific log_dir
9. [utils.py](utils.py): Contains following general utility functions for PyTorch training:
   - `set_seeds(seed)`: Sets random seeds for torch operations
   - `walk_through_dir(dir_path)` : Walks through dir_path returning its contents
   - `download_data(source, destination, remove_source`: Downloads a zipped dataset from source and unzips to destination.
