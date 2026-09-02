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


def _normalize_state_dict_key(key: str) -> str:
    """Normalize layer keys by their numeric index and final tensor attribute."""
    if not isinstance(key, str):
        return str(key)

    normalized = key.lower().replace('module.', '').replace('backbone.', '')
    normalized = normalized.replace('features_', 'features.')
    normalized = normalized.replace('classifier_', 'classifier.')
    normalized = normalized.replace('bn_', 'bn.')
    normalized = normalized.replace('running_mean', 'running.mean')
    normalized = normalized.replace('running_var', 'running.var')
    normalized = normalized.replace('num_batches_tracked', 'num.batches.tracked')
    normalized = normalized.replace('_', '.')

    if normalized.startswith('_initializer_'):
        return normalized

    match = re.search(r'(\d+)(?:\.(.*))?$', normalized)
    if not match:
        return normalized

    layer_idx = match.group(1)
    suffix = (match.group(2) or '').strip('.')
    if suffix:
        return f"{layer_idx}.{suffix}"
    return layer_idx


def _match_model_state_dict(model: torch.nn.Module, pretrained_weights: dict) -> dict:
    """Map checkpoint tensors to model keys using a name match, then a shape-based fallback for renamed ONNX/exported checkpoints."""
    if not isinstance(pretrained_weights, dict):
        return {}

    model_state = model.state_dict()
    checkpoint_items = list(enumerate(pretrained_weights.items()))
    checkpoint_aliases = {}
    for idx, (checkpoint_key, tensor) in checkpoint_items:
        normalized = _normalize_state_dict_key(checkpoint_key)
        checkpoint_aliases.setdefault(normalized, []).append((idx, checkpoint_key, tensor))

    matched = {}
    used_indices = set()

    for model_key, model_tensor in model_state.items():
        model_aliases = {
            _normalize_state_dict_key(model_key),
            model_key.lower().replace('backbone.', ''),
            model_key.lower().split('.')[-1],
        }

        for alias in model_aliases:
            candidates = checkpoint_aliases.get(alias, [])
            for idx, checkpoint_key, checkpoint_tensor in candidates:
                if idx in used_indices:
                    continue
                if checkpoint_tensor.shape == model_tensor.shape:
                    matched[model_key] = checkpoint_tensor
                    used_indices.add(idx)
                    break
            if model_key in matched:
                break

    for model_key, model_tensor in model_state.items():
        if model_key in matched:
            continue
        for idx, (checkpoint_key, checkpoint_tensor) in checkpoint_items:
            if idx in used_indices:
                continue
            if checkpoint_tensor.shape == model_tensor.shape:
                matched[model_key] = checkpoint_tensor
                used_indices.add(idx)
                break

    return matched



def load_pretrained_weights(model: torch.nn.Module, model_name: str, display: bool = True) -> Tuple[torch.nn.Module, Optional[str]]:
    """Load pretrained weights into the model, ignoring mismatched layers."""
    weights_folder = os.path.join(project_root, "models", "weights")
    if not os.path.exists(weights_folder):
        print(f"Warning: Weights folder '{weights_folder}' does not exist. Skipping weight loading.") if display else None
        return model, "unknown"

    weights_list = os.listdir(weights_folder)
    weight_file = next((file for file in weights_list if model_name.lower() in file.lower()), None)

    if weight_file is None:
        print(f"No pretrained weights found for model: {model_name}") if display else None
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
    pretrained_dict = _match_model_state_dict(model, pretrained_weights)

    if not pretrained_dict:
        print(
            f"Warning: no compatible weight keys matched for model '{model_name}' in '{file_path}'. "
            "This usually means the checkpoint was trained with a different architecture, a different module naming scheme, "
            "or an ONNX export that uses generic initializer names such as '_initializer_123'."
        ) if display else None
        return model, dataset_name

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers from {file_path} for model: {model_name}") if display else None
    return model, dataset_name


def load_model(
        model_name: str, 
        num_channels: int = 3, 
        num_outputs: int = 1, 
        dropout_rate: float = 0.3, 
        freezed: bool = False,
        display: bool = True
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
    print(f"Model: {model_name} - Trainable params: {trainable_params} / {total_params}") if display else None
    if trainable_params == 0:
        print("WARNING: No trainable parameters! Check model freezing logic.") if display else None

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


def compute_batch_loss(
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        weights: torch.Tensor, 
        criterion_type: str = "mse",
        alpha: float = 0.5
    ) -> torch.Tensor:
    """Compute the batch loss based on the specified criterion type."""
    mse_criterion: torch.nn.Module = torch.nn.MSELoss(reduction='none')
    ccc_criterion: torch.nn.Module = CCCLoss()

    if criterion_type == "mse":
        per_dim_loss = mse_criterion(outputs, targets).mean(dim=0)  # Mean over the batch for each dimension
    elif criterion_type == "ccc":
        per_dim_loss = ccc_criterion(outputs, targets)
    elif criterion_type == "combined":
        per_dim_mse_loss = mse_criterion(outputs, targets).mean(dim=0)
        per_dim_ccc_loss = ccc_criterion(outputs, targets)
        per_dim_loss = alpha * per_dim_mse_loss + (1 - alpha) * per_dim_ccc_loss
    else:
        raise ValueError(f"Unsupported criterion type: {criterion_type}. Choose from 'mse', 'ccc', or 'combined'.")

    per_dim_loss = per_dim_loss.view(-1)  # Ensure per_dim_loss is a 1D tensor
    weights = weights.view(-1)  # Ensure weights is a 1D tensor

    weighted_loss = torch.sum(per_dim_loss * weights, dim=0) / torch.sum(weights)
    return weighted_loss

class CCCLoss(torch.nn.Module):
    """Concordance Correlation Coefficient (CCC) Loss for regression tasks."""
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps  # Small epsilon to avoid division by zero

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the CCC loss between predictions and targets."""
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        mean_preds = torch.mean(preds, dim=0)
        mean_targets = torch.mean(targets, dim=0)

        var_preds = torch.var(preds, dim=0, unbiased=False)
        var_targets = torch.var(targets, dim=0, unbiased=False)

        covariance = torch.mean((preds - mean_preds) * (targets - mean_targets), dim=0)

        ccc = (2.0 * covariance) / (var_preds + var_targets + (mean_preds - mean_targets) ** 2 + 1e-8)
        return 1.0 - ccc  # Return 1 - CCC as the loss to minimize











