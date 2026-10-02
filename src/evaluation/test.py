"""Evaluation script for VAD model testing and metrics computation."""

import argparse
from typing import Dict, List, Tuple, Union
from pathlib import Path

import numpy as np
import torch

from src.utils import VADDataset, load_model, get_transforms, compute_metrics, get_simu_params

from src.utils import (
    VADDataset,
    VADLoss,
    compute_metrics,
    get_simu_params,
    get_transforms,
    load_model,
    get_project_root,    
)


def evaluate(
        dataloader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: VADLoss,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        display: bool = True,
    ) -> tuple[float, dict[str, float]]:
    """Evaluates a model on the provided dataloader.

    Args:
        dataloader: PyTorch DataLoader containing evaluation dataset.
        model: PyTorch model instance to evaluate.
        criterion: VADLoss criterion function.
        device: Calculation device (CPU or CUDA).
        display: Whether to print visual progress bar.

    Returns:
        Tuple of (average_loss, metrics_dictionary).
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    total_batches = len(dataloader)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader, 1):
            progress = (i / total_batches) 
            bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
            print(f"Evaluation: |{bar}| {progress * 100 :.2f}% [{i}/{total_batches}]", end="\r") if display else None

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if outputs.shape != targets.shape:
                raise ValueError(f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}")

            batch_loss = criterion(outputs, targets)
            total_loss += batch_loss.item()
            
            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    print(" " * 80, end="\r") if display else None  # Clear the progress bar line

    avg_loss = total_loss / max(total_batches, 1)
    metrics = compute_metrics(torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0))
    return avg_loss, metrics


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the evaluation script."""
    parser = argparse.ArgumentParser(description="VAD Evaluation (MSE & RMSE per dimension).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device to use for evaluation (default: cuda if available, otherwise cpu).")
    parser.add_argument("--split", type=str, default="Test", choices=["Test", "Val", "Train"], help="Data split to evaluate on (default: Test).")
    parser.add_argument("--state_dict_dir", type=str, required=True, help="Path from /output/ to dir containing the state dict with weights (.pth).")
    return parser

def main() -> None:
    """CLI entry point for the evaluation script."""
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    state_dict_path = Path(args.state_dict_dir) / "best_model_state.pth"
    if not state_dict_path.is_absolute():
        state_dict_path = Path(get_project_root()) / "output" / state_dict_path

    if not state_dict_path.exists():
        raise FileNotFoundError(f"State dict path does not exist: {state_dict_path}")
    
    # Extract simulation parameters from the directory name of the state_dict_path
    simu_params = get_simu_params(state_dict_path)
    dataset_name = simu_params.get("dataset", "fer")
    dropout_rate = float(simu_params.get("dropout", 0.5))
    batch_size = int(simu_params.get("bs", 64))
    modelweights = simu_params.get("weights", None)
    input_size = int(simu_params.get("size", 112))
    criterion_type = simu_params.get("criterion", "mse")
    ccc_weight = float(simu_params.get("cccweight", 0.0))
    num_channels = 1 if modelweights == "custom" else 3

    model_name = state_dict_path.parts[-3] if len(state_dict_path.parts) >=3 else "resnet50"  # Model architecture name inferred from the directory structure

    # Determine active VAD dimension flags and weights
    raw_weights = [float(simu_params.get("V", 1.0)), float(simu_params.get("A", 1.0)), float(simu_params.get("D", 1.0))]
    if dataset_name == "afew":
        raw_weights[2] = 0.0  # Set Dominance weight to 0 for AFEW dataset

    target_names = ["Valence", "Arousal", "Dominance"]
    include_flags =  [w > 0 for w in raw_weights]
    active_weights = [w for w, include in zip(raw_weights, include_flags) if include]
    num_outputs = len(active_weights)
    weights_tensor = torch.tensor(active_weights, dtype=torch.float32, device=device)

    # Instantiate model architecture and loss criterion 
    if criterion_type == "combined":
        criterion = VADLoss(weights=weights_tensor, alpha=ccc_weight)
    else:
        criterion = VADLoss(weights=weights_tensor, alpha=0.0)

    print(f"Instantiating model {model_name.upper()} with {num_outputs} outputs and dropout rate {dropout_rate}")
    model = load_model(
        model_name=model_name,
        num_channels=num_channels,
        num_outputs=num_outputs,
        dropout_rate=dropout_rate
    )

    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.to(device)

    # Setup image normalization transforms based on pretrained source
    if modelweights == "custom":
        image_mean = [0.5] * num_channels
        image_std = [0.5] * num_channels
    elif modelweights == "imagenet":
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
    else:
        image_mean = [0.5] * num_channels
        image_std = [0.5] * num_channels

    _, test_transform = get_transforms(image_mean=image_mean, image_std=image_std, input_size=input_size)

    # Build dataset and dataloader for evaluation
    dataset = VADDataset(
        dataset=dataset_name, 
        split=args.split, 
        transform=test_transform, 
        include_V=include_flags[0], 
        include_A=include_flags[1], 
        include_D=include_flags[2],
        num_channels=num_channels,
        display=True
    )

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Run evaluation using checkpoint loss criterion & alpha
    print(f"Evaluating model on  dataset {dataset_name} (split: {args.split}) with batch size {batch_size}...")
    avg_loss, metrics = evaluate(
        test_loader, 
        model, 
        criterion=criterion,
        device=device
    )

    # Display results
    active_target_names = [n for n, inc in zip(target_names, include_flags) if inc]
    print(f"\n" + "=" * 50)
    print(f"EVALUATION RESULTS ({args.split.upper()} SPLIT)")
    print("=" * 50)
    print(f"Average Loss ({criterion_type.upper()}): {avg_loss:.4f}\n")

    print(f"Overall Metrics:")
    print(f"  MSE  : {metrics.get('mse_overall', 'N/A'):.4f}")
    print(f"  RMSE : {metrics.get('rmse_overall', 'N/A'):.4f}")
    print(f"  CCC  : {metrics.get('ccc_overall', 'N/A'):.4f}\n")

    print("Per-Dimension Metrics:")
    rmse_per_dim = metrics.get("rmse_per_dim", [])
    ccc_per_dim = metrics.get("ccc_per_dim", [])
    for idx, name in enumerate(active_target_names):
        r_val = f"{rmse_per_dim[idx]:.4f}" if idx < len(rmse_per_dim) else "N/A"
        c_val = f"{ccc_per_dim[idx]:.4f}" if idx < len(ccc_per_dim) else "N/A"
        print(f"  {name:<10} -> RMSE: {r_val} | CCC: {c_val}")


if __name__ == "__main__":
    main()