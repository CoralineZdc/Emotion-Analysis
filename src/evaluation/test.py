import os
import sys
import argparse
from typing import List
import numpy as np
import torch
from pathlib import Path
import optuna
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize

from src.utils.data_loader import DataLoader
from src.utils.training_utils import load_model, compute_batch_loss, get_transforms, compute_metrics 
from src.utils.parsing_utils import get_simu_params


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate(
        dataloader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        weights: torch.Tensor,
        criterion_type: str = "mse",
        alpha: float = 0.5,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        trial: optuna.trial.Trial | None = None
    ) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    total_batches = len(dataloader)
    is_optuna = trial is not None

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader, 1):
            progress = (i / total_batches) 
            bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
            print(f"Evaluation: |{bar}| {progress * 100 :.2f}% [{i}/{total_batches}]", end="\r") if not is_optuna else None

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if outputs.shape != targets.shape:
                raise ValueError(f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}")

            batch_loss = compute_batch_loss(outputs, targets, weights, criterion_type, alpha)
            total_loss += batch_loss.item()
            
            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    print(" " * 80, end="\r") if not is_optuna else None  # Clear the progress bar line

    avg_loss = total_loss / max(total_batches, 1)
    metrics = compute_metrics(torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0))
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="VAD Evaluation (MSE & RMSE per dimension).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device to use for evaluation (default: cuda if available, otherwise cpu).")
    parser.add_argument("--split", type=str, default="Test", choices=["Test", "Val", "Train"], help="Data split to evaluate on (default: Test).")
    parser.add_argument("--state_dict_path", type=str, required=True, help="Path to state dict with weights (.pth).")

    args = parser.parse_args()
    device = torch.device(args.device)

    state_dict_path = Path(args.state_dict_path)
    if not state_dict_path.is_absolute():
        state_dict_path = Path(repo_root()) / state_dict_path

    simu_params = get_simu_params(state_dict_path)
    dataset_name = simu_params.get("dataset", "fer")
    dropout_rate = float(simu_params.get("dropout", 0.5))
    batch_size = int(simu_params.get("bs", 64))
    modelweights = simu_params.get("weights", None)
    input_size = int(simu_params.get("size", 112))
    criterion = simu_params.get("criterion", "mse")
    ccc_weight = float(simu_params.get("cccweight", 0.0))

    model_name = state_dict_path.parts[-3] if len(state_dict_path.parts) >=3 else "resnet50"  # Assuming the model name is the third last part of the path

    num_channels = 1 if modelweights == "custom" else 3
    DataLoader.set_num_channels(num_channels)

    # Initialize DataLoader protocol to ensure image and label statistics are computed
    DataLoader.set_data_protocol("small_split")
    DataLoader._ensure_image_stats(dataset_name)
    DataLoader._ensure_label_stats(dataset_name)
    print(f"Evaluating model on {dataset_name} dataset")

    # Determine which dimensions to include based on weights and dataset availability
    weights_list = [float(simu_params.get("V", 1.0)), float(simu_params.get("A", 1.0)), float(simu_params.get("D", 1.0))]
    include_flags = [(w > 0 and dim in DataLoader.available_columns) for w, dim in zip(weights_list, ["Valence", "Arousal", "Dominance"])]
    num_outputs = sum(include_flags)

    weights_list = [w for w, include in zip(weights_list, include_flags) if include]
    weights = torch.tensor(weights_list, dtype=torch.float32, device=device)

    #Instanciate model
    print(f"Instantiating model {model_name.upper()} with {num_outputs} outputs and dropout rate {dropout_rate}")
    model = load_model(
        model_name=model_name,
        num_channels=num_channels,
        num_outputs=num_outputs,
        dropout_rate=dropout_rate
    )

    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.to(device)

    # Determine image normalization parameters based on whether the model is pretrained or not
    if modelweights == "custom":
        image_mean = torch.tensor([0.5], dtype=torch.float32, device=device)
        image_std = torch.tensor([0.5], dtype=torch.float32, device=device)
    elif modelweights == "imagenet":
        image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)
        image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)
    else:
        image_mean = DataLoader.image_mean
        image_std = DataLoader.image_std
        if isinstance(image_mean, (torch.Tensor, np.ndarray)):
            image_mean = image_mean.tolist()
        if isinstance(image_std, (torch.Tensor, np.ndarray)):
            image_std = image_std.tolist()

    # Evaluation transforms and loader

    _, test_transform = get_transforms(image_mean=image_mean, image_std=image_std, input_size=input_size)

    dataset = DataLoader(
        dataset=dataset_name, 
        split=args.split, 
        transform=test_transform, 
        include_V=include_flags[0], 
        include_A=include_flags[1], 
        include_D=include_flags[2])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Run evaluation using checkpoint loss criterion & alpha
    avg_loss, metrics = evaluate(
        test_loader, 
        model, 
        weights=weights, 
        criterion_type=criterion, 
        alpha=ccc_weight, 
        device=device
    )

    # Display results
    print(f"\n=== Evaluation Results on '{args.split}' ===")
    print(f"Average Loss: {avg_loss:.4f}\n")

    for metric, value in metrics.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            formatted_val = ", ".join(f"{v:.4f}" for v in value)
            print(f"{metric:<20} | [{formatted_val}]")
        elif isinstance(value, (float, int)):
            print(f"{metric:<20} | {value:<10.4f}")
        else:
            print(f"{metric:<20} | {str(value)}")


if __name__ == "__main__":
    main()