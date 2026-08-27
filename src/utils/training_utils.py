import numpy as np
import random
import torch
import os
import onnx2pytorch
import onnx
import re
from torch.nn import functional as F
import numpy as np
from typing import Optional, Tuple

from models import resnet, vgg, mobilenet, mobilefacenet, efficientnet

# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and cuDNN for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_gradient(optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    """Clip gradients computed during backpropagation to avoid explosion of gradients."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def load_pretrained_weights(model: torch.nn.Module, model_name: str) -> Tuple[torch.nn.Module, Optional[str]]:
    """Load pretrained weights into the model, ignoring mismatched layers."""
    weights_folder = os.path.join(project_root, "models", "weights")
    if not os.path.exists(weights_folder):
        print(f"Warning: Weights folder '{weights_folder}' does not exist. Skipping weight loading.")
        return model, None

    weights_list = os.listdir(weights_folder)
    weight_file = next((file for file in weights_list if model_name.lower() in file.lower()), None)

    if weight_file is None:
        print(f"No pretrained weights found for model: {model_name}")
        return model, "unknown"  # Return the model without loading weights

    file_path = os.path.join(weights_folder, weight_file)
    file_name = os.path.basename(weight_file)
    root, extension = os.path.splitext(file_name)

    parts = root.split("_")
    dataset_name = parts[1] if len(parts) > 1 else None  # Assuming the format is model_dataset.pth or model_dataset.pt

    if extension in [".pth", ".pt"]:
        pretrained_weights = torch.load(file_path, map_location=torch.device('cpu'))
    elif extension == ".onnx":
        onnx_model = onnx.load(file_path)
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
        pretrained_weights = pytorch_model.state_dict()
    else:
        raise ValueError(f"Unsupported weight file format: {extension}")

    if isinstance(pretrained_weights, dict) and 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers from {file_path} for model: {model_name}")
    return model, dataset_name


def load_model(
        model_name: str, 
        num_channels: int = 3, 
        num_outputs: int = 1, 
        dropout_rate: float = 0.3, 
        freezed: bool = False
    ) -> torch.nn.Module:
    """Instantiate a model based on the specified architecture and parameters."""
    MODEL_CLASSES = {
        "resnet": resnet.ResNetRegression,
        "vgg": vgg.VGGRegression,
        "mobilenet": mobilenet.MobileNet,
        "mobilefacenet": mobilefacenet.MobileFaceNet,
        "efficientnet": efficientnet.EfficientNetB0
    }

    base_name = re.findall(r'[a-zA-Z]+', model_name)[0].lower() if "resnet" in model_name or "vgg" in model_name else model_name.lower()
    model_class = MODEL_CLASSES.get(base_name)

    if model_class is None:
        raise ValueError(f"Unsupported model architecture: {model_name}. Available options are: {list(MODEL_CLASSES.keys())}")

    if base_name in ["resnet", "vgg"]:
        model = model_class(
            model_name=model_name, 
            num_channels=num_channels, 
            num_outputs=num_outputs, 
            dropout_rate=dropout_rate, 
            freezed=freezed)
    else:
        model = model_class(
            num_channels=num_channels, 
            num_outputs=num_outputs, 
            dropout_rate=dropout_rate, 
            freezed=freezed)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} - Trainable params: {trainable_params} / {total_params}")
    if trainable_params == 0:
        print("WARNING: No trainable parameters! Check model freezing logic.")

    return model

    
def orth_dist(weight: torch.Tensor) -> torch.Tensor:
    """
    Computes soft orthogonality loss across matrix/tensor dimensions:
    Loss = || W W^T - I ||_F^2 (or || W^T W - I ||_F^2 depending on matrix shape).
    """
    if weight.dim() == 4: # Convolutional layer weights
        w_flat = weight.view(weight.size(0), -1)  # Flatten to (out_channels, in_channels * kernel_height * kernel_width)
    elif weight.dim() == 2: # Fully connected layer weights
        w_flat = weight  # Already in the correct shape
    else:
        return torch.tensor(0.0, device=weight.device)  # No orthogonality loss for other dimensions

    rows, cols = w_flat.size()
    if rows <= cols:
        gram = torch.mm(w_flat, w_flat.t())
        identity = torch.eye(rows, device=weight.device)
    else:
        gram = torch.mm(w_flat.t(), w_flat)
        identity = torch.eye(cols, device=weight.device)

    return torch.norm(gram - identity, p='fro')  # Frobenius norm of the difference


def conv_orth_loss(layer: torch.nn.Conv2d) -> torch.Tensor:
    """
    Computes orthogonality loss for a convolutional layer by enforcing spatial orthogonality.
    This is done by convolving the kernel with itself and comparing it to a Dirac delta function.
    Loss = || Conv(K, K, padding=P, stride=S) - I ||_F^2
    """
    kernel = layer.weight
    c_out, c_in, h, w = kernel.shape

    if c_out != c_in or h != w:
        return orth_dist(kernel)  # Fallback to standard orthogonality loss if not square

    try:
        conv_output = F.conv2d(kernel, kernel, stride=layer.stride, padding=layer.padding)
        h_out, w_out = conv_output.shape[-2:]

        target = torch.zeros_like(conv_output)
        cy, cx = h_out // 2, w_out // 2
        target[:, :, cy, cx] = torch.eye(c_out, device=kernel.device)

        return torch.sum((conv_output - target) ** 2)  # Frobenius norm of the difference || Conv(K, K, padding=P, stride=S) - I ||_F^2

    except RuntimeError:
        return orth_dist(kernel)  # Fallback to standard orthogonality loss if convolution fails


def compute_orth_loss_model(model: torch.nn.Module) -> torch.Tensor:
    """
    Generalized function to compute orthogonality loss for all convolutional and linear layers in a model.
    This function iterates through the model's parameters, identifies convolutional and linear layers,
    and computes the orthogonality loss for each layer. The total loss is the sum of individual losses.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)  # Initialize loss on the same device as model parameters
    count = 0  # Counter for the number of layers contributing to the loss

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.weight.requires_grad:
            loss += conv_orth_loss(module)
            count += 1
        elif isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
            loss += orth_dist(module.weight)
            count += 1

    return loss / max(count, 1)  # Return average loss to avoid division by zero


def compute_weighted_loss(
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        weights: torch.Tensor, 
        criterion: torch.nn.Module
    ) -> torch.Tensor:
    """Compute the weighted across active dimensions."""
    loss_per_dim = criterion(outputs, targets)
    normalized_weights = weights / weights.sum()
    weighted_loss = loss_per_dim * normalized_weights
    return weighted_loss.sum(dim=1).mean()


def compute_unnormlized_rmse(
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        label_mean: torch.Tensor, 
        label_std: torch.Tensor
    ) -> np.ndarray:
    """Calculate the unnormalized RMSE for each dimension."""
    preds_raw = preds * label_std + label_mean
    targets_raw = targets * label_std + label_mean
    rmse_per_dim = torch.sqrt(torch.mean((preds_raw - targets_raw) ** 2, dim=0)).cpu().numpy()
    return rmse_per_dim











