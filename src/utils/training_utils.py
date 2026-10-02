"""Training utility module providing model loading, loss criteria, and metrics."""

import random
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnx2pytorch
import torchvision.models as tv_models
from torchvision.transforms import v2 as transforms

from models import efficientnet, mobilefacenet, mobilenet, resnet, vgg
from src.utils.parsing_utils import get_project_root


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


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Sets gradient computation status for backbone model layers."""
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = trainable
    else:
        for name, param in model.named_parameters():
            if "head" not in name.lower():
                param.requires_grad = trainable


def get_parameter_groups(
        model: torch.nn.Module, 
        head_lr: float, 
        backbone_lr: float, 
        weight_decay: float
    ) -> List[dict]:
    """
    Constructs optimizer parameter groups for differential learning rates.
    
    Args:
        model: PyTorch model instance.
        head_lr: Learning rate for the model's head.
        backbone_lr: Learning rate for the model's backbone.
        weight_decay: Weight decay for regularization.
    Returns:
        List of parameter groups for the optimizer.
    """
    if hasattr(model, "head") and hasattr(model, "backbone"):
        head_params = list(model.head.parameters())
        backbone_params = list(model.backbone.parameters())
    else:
        head_params = [p for n, p in model.named_parameters() if "head" in n.lower()]
        backbone_params = [p for n, p in model.named_parameters() if "head" not in n.lower()]

    return [
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "head"},
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"}
    ]


def get_transforms(
        image_mean: list, image_std: list, input_size: int = 224, data_augmentation: bool = False
    ) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train and validation image transform pipelines.
    
    Args:
        image_mean: Mean values for image normalization.
        image_std: Standard deviation values for image normalization.
        input_size: Size of the input images.
        data_augmentation: Whether to apply data augmentation.
    Returns:
        Tuple of train and validation image transform pipelines.
    """
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    if not data_augmentation:
        return val_transform, val_transform

    pad = int(input_size * 0.15)
    train_transform = transforms.Compose([
        transforms.Resize((input_size + pad, input_size + pad)),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10.0, 10.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=image_mean, std=image_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
    ])

    return train_transform, val_transform


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient (CCC) Loss for regression tasks."""
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps 

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the CCC loss between predictions and targets."""
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        mean_preds, mean_targets = torch.mean(preds, dim=0), torch.mean(targets, dim=0)
        var_preds, var_targets = torch.var(preds, dim=0, unbiased=False), torch.var(targets, dim=0, unbiased=False)

        cov = torch.mean((preds - mean_preds) * (targets - mean_targets), dim=0)
        ccc = (2.0 * cov) / torch.clamp(var_preds + var_targets + (mean_preds - mean_targets) ** 2 + 1e-8, min=1e-6)
        return 1.0 - ccc 


class VADLoss(nn.Module):
    """Unified loss module supporting MSE, CCC, and Combined Losses."""

    def __init__(self, criterion_type: str = "mse", weights: torch.Tensor = torch.tensor([1.0, 1.0, 1.0]), alpha: float = 0.5) -> None:
        super().__init__()
        self.criterion_type = criterion_type.lower()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")
        self.ccc = CCCLoss()
        self.weights = weights

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the batch loss based on the specified criterion type.
        
        Args:
            preds: Model predictions (batch_size x num_dimensions).
            targets: Ground truth targets (batch_size x num_dimensions).
        Returns:
            Weighted loss scalar for the batch.
        """
        if self.criterion_type == "mse":
            per_dim_loss = self.mse(preds, targets).mean(dim=0)  # Mean over the batch for each dimension
        elif self.criterion_type == "ccc":
            per_dim_loss = self.ccc(preds, targets)
        elif self.criterion_type == "combined":
            per_dim_mse_loss = self.mse(preds, targets).mean(dim=0)
            per_dim_ccc_loss = self.ccc(preds, targets)
            per_dim_loss = self.alpha * per_dim_ccc_loss + (1 - self.alpha) * per_dim_mse_loss
        else:
            raise ValueError(f"Unsupported criterion type: {self.criterion_type}. Choose from 'mse', 'ccc', or 'combined'.")

        per_dim_loss = per_dim_loss.view(-1)
        weights = self.weights.view(-1)
        weighted_loss = torch.sum(per_dim_loss * weights, dim=0) / torch.sum(weights)
        return weighted_loss


def load_model(
        model_name: str, 
        num_channels: int = 3, 
        num_outputs: int = 1, 
        dropout_rate: float = 0.3, 
        freezed: bool = False,
        display: bool = True
    ) -> torch.nn.Module:
    """
    Instantiate a model based on the specified architecture and parameters.

    Args:
        model_name: Name of the model to instantiate.
        num_channels: Number of input channels.
        num_outputs: Number of output dimensions.
        dropout_rate: Dropout rate for the model.
        freezed: Whether to freeze the model's parameters.
        display: Whether to display model information.
    Returns:
        The instantiated model.
    """
    model_classes = {
        "resnet": resnet.ResNetRegression,
        "vgg": vgg.VGGRegression,
        "mobilenet": mobilenet.MobileNet,
        "mobilefacenet": mobilefacenet.MobileFaceNet,
        "efficientnet": efficientnet.EfficientNetB0
    }

    base_name = re.findall(r'[a-zA-Z]+', model_name)[0].lower() if "resnet" in model_name or "vgg" in model_name else model_name.lower()
    model_class = model_classes.get(base_name)

    if model_class is None:
        raise ValueError(f"Unsupported model architecture: {model_name}. Available options are: {list(model_classes.keys())}")

    kwargs = {
        "num_channels": num_channels,
        "num_outputs": num_outputs,
        "dropout_rate": dropout_rate,
        "freezed": freezed,
    }
    if base_name in ["resnet", "vgg"]:
        kwargs["model_name"] = model_name

    model = model_class(**kwargs)

    if display:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_name} - Trainable params: {trainable_params} / {total_params}")
        if trainable_params == 0:
            print("WARNING: No trainable parameters! Check model freezing logic.")

    return model


def load_pretrained_weights(
        model: torch.nn.Module, 
        model_name: str, 
        weights_source: str = "imagenet",
        display: bool = True
    ) -> Tuple[torch.nn.Module, Optional[str]]:
    """
    Loads pre-trained weights into the target model architecture.
    
    Args:
        model: The target PyTorch model instance.
        model_name: Name of the model architecture.
        weights_source: Source of the weights ('imagenet' or 'custom').
        display: Whether to print loading information.
    Returns:
        Tuple of (model with loaded weights, source of weights or None).
    """
    if weights_source == "imagenet":
        if model_name == "efficientnet":
            tv_model_name = "efficientnet_b0"
        elif model_name == "mobilenet":
            tv_model_name = "mobilenet_v2"
        else:
            tv_model_name = model_name
        try:
            tv_model = tv_models.get_model(tv_model_name, weights="DEFAULT")
            model.load_state_dict(tv_model.state_dict(), strict=False)
            return model, "imagenet"
        except Exception as e:
            if display:
                print(f"Error loading torchvision weights for {model_name}: {e}")
            return model, "unknown"

    elif weights_source == "custom":
        # Load weights from the 'models/weights' directory, matching by model name
        weights_dir = get_project_root() / "models" / "weights"
        if not weights_dir.exists():
            if display:
                print(f"Warning: Weights folder '{weights_dir}' does not exist. Skipping weight loading.")
            return model, "unknown"

        match_file = next((file for file in weights_dir.iterdir() if model_name.lower() in file.name.lower()), None)
        if not match_file:
            if display:
                print(f"No pretrained weights found for model: {model_name}")
            return model, "unknown"

        # Determine the file extension and load accordingly
        extension = match_file.suffix.lower()
        if extension in [".pth", ".pt"]:
            raw_state_dict = torch.load(match_file, map_location=torch.device('cpu'))
        elif extension == ".onnx":
            onnx_model = onnx.load(match_file)
            pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
            raw_state_dict = pytorch_model.state_dict()
        else:
            raise ValueError(f"Unsupported weight file format: {extension}")

        if raw_state_dict is None or not isinstance(raw_state_dict, dict):
            if display:
                print(f"Warning: No valid state_dict found in {match_file}. Skipping weight loading.")
            return model, "unknown"

        # Handle common nested keys in state_dicts (e.g., 'state_dict', 'model', 'net')
        for key in ["state_dict", "model", "net"]:
            if key in raw_state_dict and isinstance(raw_state_dict[key], dict):
                raw_state_dict = raw_state_dict[key]
                break

        # Match and load compatible weights into the model
        matched_dict = _match_model_state_dict(model, raw_state_dict)
        if not matched_dict:
            if display:
                print(f"Warning: No compatible weight keys matched for model '{model_name}' in '{match_file}'.")
            return model, "unknown"

        model_dict = model.state_dict()
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict, strict=False)

        if display:
            print(f"Successfully loaded {len(matched_dict)}/{len(model_dict)} layers from {match_file} for model: {model_name}")
            
    else:
        raise ValueError(f"Unsupported weights source: {weights_source}. Choose 'imagenet' or 'custom'.")
    return model, "unknown"


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
    Match and adapt keys from a raw state_dict to the target model's state_dict.

    Args:
        model: The target PyTorch model instance.
        raw_state_dict: The source state_dict to match against.
    Returns:
        A dictionary of matched and adapted weights for the model.    
    """
    if not isinstance(raw_state_dict, dict):
        return {}

    model_state = model.state_dict()
    matched = {}
    used_raw_keys = set()

    for raw_key, src_tensor in raw_state_dict.items():
        # Clean the raw key to remove common prefixes and ONNX graph paths
        clean_key = _clean_onnx_prefix(raw_key)
        if any(ignored in clean_key.lower() for ignored in ["head.", "fc.", "classifier.6", "output."]):
            continue

        # Generate candidate keys to match against the model's state_dict
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

    unmatched_model_keys = [
        k for k in model_state.keys() 
        if k not in matched and not any(h in k for h in ["head.", "classifier.6"])
    ]
    
    unused_raw_items = [
        (k, v) for k, v in raw_state_dict.items() 
        if k not in used_raw_keys and not any(h in k.lower() for h in ["fc", "head", "output"])
    ]

    # Try to match any remaining unmatched model keys with unused raw items
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


def apply_mixup(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2):
    """Applies Mixup interpolation to input images and VAD targets."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]
    
    return mixed_inputs, mixed_targets