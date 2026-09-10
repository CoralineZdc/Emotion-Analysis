import os
import sys
import argparse
from typing import List
import numpy as np
import torch
from pathlib import Path
import optuna
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize


# Navigate UP 3 levels: evaluation -> src -> Project Root
project_root = Path(__file__).resolve().parents[2]
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.data_loader import DataLoader
from src.utils.training_utils import load_model, compute_batch_loss


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_simu_params(state_dict_path):
    """Extracts simulation parameters from the directory name of the state_dict_path."""
    dir_name = state_dict_path.parent.name
    params = {}
    for param in dir_name.split("_"):
        if "-" in param:
            key, value = param.split("-", 1)
            params[key] = value
        else:
            idx = next((i for i, c in enumerate(param) if c.isdigit()), len(param))
            key, value = param[:idx], param[idx:]
            if key:
                params[key] = value
    return params


def evaluate(
        dataloader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion_type: str = "mse",
        alpha: float = 0.5,
        weights: torch.Tensor | None = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        trial: optuna.trial.Trial | None = None
    ) -> tuple[float, np.ndarray]:
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

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    rmse_per_dim = torch.sqrt(torch.mean((preds_cat - targets_cat) ** 2, dim=0)).cpu().numpy()

    return avg_loss, rmse_per_dim


def main():
    parser = argparse.ArgumentParser(description="VAD Evaluation (MSE & RMSE per dimension).")
    parser.add_argument("--input_size", type=int, default=48, help="Image spatial resolution (default: 48).")
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
    batch_size = int(simu_params.get("batch", 64))
    modelweights = simu_params.get("modelweights", None)

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
        image_mean = DataLoader.image_mean.to(device, dtype=torch.float32)
        image_std = DataLoader.image_std.to(device, dtype=torch.float32)

    # Evaluation transforms and loader
    test_transform = Compose([
        Resize((args.input_size, args.input_size)),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=image_mean.tolist(), std=image_std.tolist())
    ])

    dataset = DataLoader(
        split=args.split, 
        dataset=dataset_name, 
        transform=test_transform, 
        include_V=include_flags[0], 
        include_A=include_flags[1], 
        include_D=include_flags[2])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Run evaluation
    avg_loss, rmse_per_dim = evaluate(test_loader, model, weights=weights, device=device)

    # Display results
    print(f"\n=== Evaluation Results on '{args.split}' ===")
    print(f"Average Loss: {avg_loss:.4f}\n")
    dims = ["Valence (V)", "Arousal (A)", "Dominance (D)"]
    dims = [dim for dim, include in zip(dims, include_flags) if include]
    print(f"{'Dimension':<15} | {'RMSE':<10}")
    print("-" * 28)
    for idx, dim_name in enumerate(dims):
        if idx < len(rmse_per_dim):
            print(f"{dim_name:<15} | {rmse_per_dim[idx]:<10.4f}")


if __name__ == "__main__":
    main()