import argparse
import csv
import os
import numpy as np
import optuna
import torch

from src.evaluation.test import evaluate
from src.utils.data_loader import DataLoader
import src.utils.transforms as transforms
from src.utils.training_utils import *


def parse_vad_weights(value: str) -> list[float]:
    """Parse three VAD loss weights from one command-line argument."""
    value = value.strip().strip("[]()")
    try:
        weights = [float(weight.strip()) for weight in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "VAD_weights must be three comma-separated numbers, e.g., 1.0,1.0,1.0"
        ) from exc

    if len(weights) != 3:
        raise argparse.ArgumentTypeError("VAD_weights must contain exactly three values: V, A, and D")
    return weights


def save_checkpoint(state: dict, filename: str) -> None:
    """Save a training checkpoint to disk."""
    torch.save(state, filename)


def train(
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    device: torch.device,
    opt: argparse.Namespace,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    trial: optuna.trial.Trial | None = None,
) -> tuple[float, np.ndarray]:
    """Run one training epoch and return the average loss and RMSE."""
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    num_batches = len(dataloader)

    # Suppress verbose batch progress printing during Optuna sweeps
    is_optuna = trial is not None
    if not is_optuna:
        print(f"\nEpoch: {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if outputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}")

        batch_loss = compute_batch_loss(outputs, targets, weights, criterion_type=opt.criterion, alpha=opt.ccc_weight)
        orth_loss = compute_orth_loss_model(model) if opt.orth_loss_weight > 0 else 0.0
        loss = batch_loss + opt.orth_loss_weight * orth_loss

        if loss.ndim > 0:
            loss = loss.mean()

        loss.backward()

        if opt.grad_clip > 0.0:
            clip_gradient(optimizer, opt.grad_clip)

        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach())
        all_targets.append(targets.detach())

        if not is_optuna:
            progress = (batch_idx + 1) / num_batches
            bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
            print(f"Training: |{bar}| {progress * 100:.2f}% [{batch_idx + 1}/{num_batches}]", end="\r")

    if not is_optuna:
        print(" " * 100, end="\r")

    avg_loss = total_loss / max(num_batches, 1)

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    rmse_per_dim = compute_unnormlized_rmse(preds_cat, targets_cat, label_mean, label_std)

    return avg_loss, rmse_per_dim



def run_training(opt: argparse.Namespace, trial: optuna.trial.Trial | None = None) -> tuple[float, np.ndarray, list[str]]:
    set_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and opt.device == "cuda" else "cpu")
    is_optuna = trial is not None

    # Fix: Dataset-specific target enforcement
    if opt.dataset.lower() == "afew":
        opt.VAD_weights = [opt.VAD_weights[0], opt.VAD_weights[1], 0.0]

    target_names = []
    active_weights = []
    for name, w in zip(["Valence", "Arousal", "Dominance"], opt.VAD_weights):
        if w > 0:
            target_names.append(name)
            active_weights.append(w)

    if not active_weights:
        raise ValueError("At least one VAD weight must be greater than 0.")

    weights_tensor = torch.tensor(active_weights, dtype=torch.float32, device=device)
    num_outputs = len(active_weights)
    include_V, include_A, include_D = opt.VAD_weights[0] > 0, opt.VAD_weights[1] > 0, opt.VAD_weights[2] > 0

    if not is_optuna:
        print(f"Starting training {opt.model} on {opt.dataset} dataset ({num_outputs} outputs)")
        print(f"Using device: {device} | Batch Size: {opt.batch_size} | Epochs: {opt.epochs}")

    # Output directory handling
    aug_str = "_data-aug" if opt.data_augmentation else ""
    ccc_str = f"_CCCweight{opt.ccc_weight:.2f}" if opt.criterion == "combined" else ""
    folder_name = (
        f"{opt.model}/seed{opt.seed}_dataset-{opt.dataset}{aug_str}_criterion-{opt.criterion}{ccc_str}"
        f"_V{opt.VAD_weights[0]:.1f}_A{opt.VAD_weights[1]:.1f}_D{opt.VAD_weights[2]:.1f}"
        f"_opt-{opt.optimizer}_lr{opt.learning_rate:.5f}_bs{opt.batch_size}_dropout{opt.dropout_rate:.1f}"
    )
    path = os.path.join(opt.output_dir, folder_name)
    os.makedirs(path, exist_ok=True)

    # Model Initialization
    # Use num_channels=1 for pretrained models (trained on grayscale), num_channels=3 for training from scratch
    num_channels = 1 if opt.pretrained else 3
    model = load_model(opt.model, num_channels=num_channels, num_outputs=num_outputs, dropout_rate=opt.dropout_rate, freezed=opt.freezed, display=not is_optuna)
    pretrain_dataset_name = None
    if opt.pretrained:
        model, pretrain_dataset_name = load_pretrained_weights(model, opt.model, display=not is_optuna)

    model.to(device)

    # Target label statistics and image statistics
    # DataLoader.size must be the size of pixels in the CSV (48x48), not the model input size
    DataLoader.size = 48 if opt.dataset == "fer" else 112
    DataLoader.set_data_protocol("small_split")
    # Set number of channels based on pretrained status: 1 for pretrained (grayscale), 3 for training from scratch
    DataLoader.set_num_channels(num_channels)
    DataLoader._ensure_image_stats(opt.dataset)
    DataLoader._ensure_label_stats(opt.dataset)

    # Ensure mean and std remain on CPU for DataLoader subprocess workers
    if pretrain_dataset_name == "ms1m":
        # Use single-channel normalization for pretrained grayscale models
        if num_channels == 1:
            image_mean = [0.5]
            image_std = [0.5]
        else:
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.5, 0.5, 0.5]
    else:
        image_mean = DataLoader.image_mean
        image_std = DataLoader.image_std
        if isinstance(image_mean, (torch.Tensor, np.ndarray)):
            image_mean = image_mean.tolist()
        if isinstance(image_std, (torch.Tensor, np.ndarray)):
            image_std = image_std.tolist()
        # Adjust to num_channels if using grayscale
        if num_channels == 1 and len(image_mean) > 1:
            image_mean = [image_mean[0]]
            image_std = [image_std[0]]

    # Transforms
    if opt.data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(opt.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((opt.input_size, opt.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((opt.input_size, opt.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    # DataLoaders with configurable parallel workers
    train_dataset = DataLoader(opt.dataset, split="Train", include_V=include_V, include_A=include_A, include_D=include_D, transform=train_transform, display=not is_optuna)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    val_dataset = DataLoader(opt.dataset, split="Val", include_V=include_V, include_A=include_A, include_D=include_D, transform=val_transform, display=not is_optuna)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    label_mean = train_dataset.active_label_mean.to(device, dtype=torch.float32)
    label_std = train_dataset.active_label_std.to(device, dtype=torch.float32)

    # Optimizer & Scheduler
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt.optimizer}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=opt.lr_factor,
        patience=opt.lr_patience,
        threshold=opt.lr_threshold,
        threshold_mode=opt.lr_threshold_mode,
        cooldown=opt.lr_cooldown,
        min_lr=opt.lr_min,
    )

    # Logging
    log_file = os.path.join(path, "log.csv")
    if not is_optuna:
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss"] + [f"{n}_train_rmse" for n in target_names] + ["val_loss"] + [f"{n}_val_rmse" for n in target_names])

    best_score = float("inf")
    best_val_rmse = None
    early_stop_counter = 0

    for epoch in range(opt.epochs):
        train_loss, train_rmse = train(epoch, trainloader, model, optimizer, weights_tensor, device, opt, label_mean, label_std, trial)
        val_loss, val_rmse = evaluate(valloader, model, opt.criterion, alpha=opt.ccc_weight, weights=weights_tensor, label_mean=label_mean, label_std=label_std, device=device, trial=trial)
        val_loss_float = float(val_loss.item()) if hasattr(val_loss, "item") else float(val_loss)

        if not is_optuna:
            print(f"Train Loss: {train_loss:.4f} | {' | '.join(f'Train RMSE {name}: {rmse:.4f}' for name, rmse in zip(target_names, train_rmse))}")
            print(f"Val Loss: {val_loss:.4f} | {' | '.join(f'Val RMSE {name}: {rmse:.4f}' for name, rmse in zip(target_names, val_rmse))}")

        scheduler.step(val_loss_float)

        if val_loss < best_score:
            best_score = val_loss
            best_val_rmse = val_rmse
            early_stop_counter = 0
            if not opt.no_model_save and not is_optuna:
                torch.save(model.state_dict(), os.path.join(path, "best_model_state.pth"))
        else:
            early_stop_counter += 1

        if early_stop_counter >= opt.early_stopping_patience:
            break

        # Optuna Intermediate Pruning Check
        if trial is not None:
            trial.report(val_loss_float, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return float(best_score), best_val_rmse, target_names


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dataset", type=str, default="fer", choices=["fer", "caers", "afew"], help ="Dataset to use for training and evaluation (default: fer)")
    parser.add_argument("--input_size", type=int, default=112, help="Image spatial resolution (default: 112)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader subprocess workers (default: 4)")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of epochs with no improvement after which training will be stopped (default: 20)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save logs and model checkpoints (default: ./output)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the optimizer (default: 5e-4)")
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "efficientnet", "mobilenet", "mobilefacenet"], help="Model architecture to use (default: vgg16)")
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained weights (default: False)")
    parser.add_argument("--freezed", action="store_true", help="Freeze the convolutional layers of the model (default: False)")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the regression head (default: 0.5)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "adamw"], help="Optimizer to use for training (default: sgd)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device to use for training (default: cuda if available, otherwise cpu)")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping value (default: 0.0, no clipping)")
    parser.add_argument("--data_augmentation", action="store_true", help="Apply data augmentation during training (default: False)")
    parser.add_argument("--resume", action="store_true", help="Resume training from a previous checkpoint (default: False)")
    parser.add_argument("--no_checkpoint", action="store_true", help="Do not save checkpoints (default: False)")
    parser.add_argument("--no_model_save", action="store_true", help="Do not save the model (default: False)")
    parser.add_argument("--criterion", type=str, default="mse", choices=["mse", "ccc", "combined"], help="Loss function to use for training (default: mse)")
    parser.add_argument("--VAD_weights", type=parse_vad_weights, default=[1.0, 1.0, 1.0], help="Weights for the VAD loss (default: [1.0, 1.0, 1.0])")
    parser.add_argument("--orth_loss_weight", type=float, default=0.5, help="Weight for the orthogonal loss (default: 0.5)")
    parser.add_argument("--ccc_weight", type=float, default=0.5, help="Weight for the CCC loss (default: 0.5)")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced (default: 0.1)")
    parser.add_argument("--lr_patience", type=int, default=10, help="Number of epochs with no improvement after which the learning rate will be reduced (default: 10)")
    parser.add_argument("--lr_threshold", type=float, default=1e-4, help="Minimum change in the monitored quantity to qualify as an improvement (default: 1e-4)")
    parser.add_argument("--lr_threshold_mode", type=str, default="rel", help="Mode to compare the monitored quantity to the threshold (default: rel)")
    parser.add_argument("--lr_cooldown", type=int, default=0, help="Number of epochs to wait before resuming normal operation after a reduction in the learning rate (default: 0)")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Lower bound on the learning rate (default: 0.0)")
    return parser


def main():
    parser = build_parser()
    opt = parser.parse_args()
    run_training(opt)


if __name__ == "__main__":
    main()