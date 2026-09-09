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
import torchvision.models as tv_models

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


def _clean_onnx_prefix(key: str) -> str:
    """Strip common wrapper and ONNX graph node prefixes from state_dict keys."""
    if not isinstance(key, str):
        return str(key)
    
    clean = key
    # Remove module/body/initializer prefixes
    for prefix in ["module.", "body.", "_initializer_", "onnx::"]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            
    # Remove leading ONNX graph paths (e.g., "/backbone/layer1/Conv" -> "layer1/Conv")
    clean = re.sub(r"^/.*?/", "", clean)
    return clean


def _adapt_channel_weights(src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Adapt 3-channel (RGB) weights to 1-channel (Grayscale) or vice-versa for input conv layers."""
    if src_tensor.shape == tgt_tensor.shape:
        return src_tensor

    # Match 4D Conv weights where only input channels (dim 1) differ
    if src_tensor.ndim == 4 and tgt_tensor.ndim == 4:
        s_out, s_in, h, w = src_tensor.shape
        t_out, t_in, th, tw = tgt_tensor.shape
        if s_out == t_out and h == th and w == tw:
            if s_in == 3 and t_in == 1:
                # Average RGB channels into 1 channel
                return src_tensor.mean(dim=1, keepdim=True)
            elif s_in == 1 and t_in == 3:
                # Repeat 1 channel across 3 RGB channels
                return src_tensor.repeat(1, 3, 1, 1) / 3.0

    return None


def _match_model_state_dict(model: torch.nn.Module, raw_state_dict: dict) -> dict:
    """
    Robustly maps checkpoint tensors to model parameters using:
    1. Name-based and prefix-aware matching.
    2. Automatic 3-channel to 1-channel input weight adaptation.
    3. Unused-pool shape-based fallback matching.
    """
    if not isinstance(raw_state_dict, dict):
        return {}

    model_state = model.state_dict()
    matched = {}
    used_raw_keys = set()

    # Pass 1: Name and prefix alignment
    for raw_key, src_tensor in raw_state_dict.items():
        clean_key = _clean_onnx_prefix(raw_key)

        # Skip final regression/classification head weights
        if any(ignored in clean_key.lower() for ignored in ["head.", "fc.", "classifier.6", "output."]):
            continue

        candidates = [
            clean_key,
            f"backbone.{clean_key}",
            clean_key.replace("backbone.", ""),
            clean_key.replace("conv_stem", "_conv_stem"),
            f"backbone.{clean_key.replace('conv_stem', '_conv_stem')}"
        ]

        for cand in candidates:
            if cand in model_state and cand not in matched:
                tgt_tensor = model_state[cand]
                adapted_tensor = _adapt_channel_weights(src_tensor, tgt_tensor)
                if adapted_tensor is not None:
                    matched[cand] = adapted_tensor
                    used_raw_keys.add(raw_key)
                    break

    # Pass 2: Unused-pool shape-based matching (Fixes index skip bug)
    unmatched_model_keys = [
        k for k in model_state.keys() 
        if k not in matched and not any(h in k for h in ["head.", "classifier.6"])
    ]
    
    unused_raw_items = [
        (k, v) for k, v in raw_state_dict.items() 
        if k not in used_raw_keys and not any(h in k.lower() for h in ["fc", "head", "output"])
    ]

    for m_key in unmatched_model_keys:
        tgt_tensor = model_state[m_key]
        for r_key, src_tensor in unused_raw_items:
            if r_key in used_raw_keys:
                continue

            adapted_tensor = _adapt_channel_weights(src_tensor, tgt_tensor)
            if adapted_tensor is not None:
                matched[m_key] = adapted_tensor
                used_raw_keys.add(r_key)
                break

    return matched

def load_pretrained_weights(
        model: torch.nn.Module, 
        model_name: str, 
        weights_source: str = "imagenet",
        display: bool = True
    ) -> Tuple[torch.nn.Module, Optional[str]]:
    """Load pretrained weights into the model, handling ONNX conversion and channel alignment."""
    raw_state_dict = None
    dataset_name = "unknown"
    source_label = ""

    if weights_source == "imagenet":
        tv_name_map = {
            "resnet18": "resnet18",
            "resnet34": "resnet34",
            "resnet50": "resnet50",
            "vgg11": "vgg11",
            "vgg13": "vgg13",
            "vgg16": "vgg16",
            "vgg19": "vgg19",
            "efficientnet": "efficientnet_b0",
            "mobilenet": "mobilenet_v2",
        }
        tv_arch = tv_name_map.get(model_name.lower(), model_name.lower())
        try:
            tv_model = tv_models.get_model(tv_arch, weights="DEFAULT")
            raw_state_dict = tv_model.state_dict()
            dataset_name = "imagenet"
            source_label = "torchvision ImageNet weights"
        except Exception as e:
            if display:
                print(f"Error loading torchvision weights for {model_name}: {e}")
            return model, "unknown"

    elif weights_source == "custom":
        weights_folder = os.path.join(project_root, "models", "weights")
        if not os.path.exists(weights_folder):
            if display:
                print(f"Warning: Weights folder '{weights_folder}' does not exist. Skipping weight loading.")
            return model, "unknown"

        weights_list = os.listdir(weights_folder)
        weight_file = next((file for file in weights_list if model_name.lower() in file.lower()), None)

        if weight_file is None:
            if display:
                print(f"No pretrained weights found for model: {model_name}")
            return model, "unknown"

        file_path = os.path.join(weights_folder, weight_file)
        source_label = f"custom weights from {file_path}"
        file_name = os.path.basename(weight_file)
        root, extension = os.path.splitext(file_name)

        parts = root.split("_")
        dataset_name = parts[1] if len(parts) > 1 else None

        if extension in [".pth", ".pt"]:
            raw_state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        elif extension == ".onnx":
            onnx_model = onnx.load(file_path)
            pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
            raw_state_dict = pytorch_model.state_dict()
        else:
            raise ValueError(f"Unsupported weight file format: {extension}")

    else:
        raise ValueError(f"Unsupported weights source: {weights_source}. Choose 'imagenet' or 'custom'.")

    if raw_state_dict is None or not isinstance(raw_state_dict, dict):
        if display:
            print(f"Warning: No valid state_dict found in {source_label}. Skipping weight loading.")
        return model, dataset_name

    # Unnest state dict wrappers if nested
    for key in ["state_dict", "model", "net"]:
        if key in raw_state_dict and isinstance(raw_state_dict[key], dict):
            raw_state_dict = raw_state_dict[key]
            break

    # Use the robust state dict matching function
    matched_dict = _match_model_state_dict(model, raw_state_dict)

    if not matched_dict:
        if display:
            print(f"Warning: No compatible weight keys matched for model '{model_name}' in '{source_label}'.")
        return model, dataset_name

    model_dict = model.state_dict()
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)

    if display:
        print(f"Successfully loaded {len(matched_dict)}/{len(model_dict)} layers from {source_label} for model: {model_name}")
        
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
            freezed=freezed
        )
    else:
        model = model_class(
            num_channels=num_channels, 
            num_outputs=num_outputs, 
            dropout_rate=dropout_rate, 
            freezed=freezed
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if display:
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


def apply_mixup(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2):
    """Applies Mixup interpolation to input images and VAD targets."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]
    
    return mixed_inputs, mixed_targets