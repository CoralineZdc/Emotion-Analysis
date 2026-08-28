import argparse
import os
import random
import numpy as np
import torch
import csv
import optuna

from src.utils.training_utils import *
from src.utils.data_loader import DataLoader
import src.utils.transforms as transforms
from src.evaluation.evaluation import evaluate


def parse_weights_vad(value: str) -> list[float]:
    """Parse three VAD loss weights from one command-line argument."""
    value = value.strip().strip("[]()")
    try:
        weights = [float(weight.strip()) for weight in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "weights_VAD must be three comma-separated numbers, "
            "for example: 1.0,1.0,1.0"
        ) from exc

    if len(weights) != 3:
        raise argparse.ArgumentTypeError(
            "weights_VAD must contain exactly three values: V, A, and D"
        )
    return weights


def save_checkpoint(
        state: dict, 
        filename: str
    ) -> None:
    """Save a training checkpoint to disk."""
    torch.save(state, filename)


def train(
        epoch: int, 
        dataloader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        weights: torch.Tensor,
        device: torch.device, 
        opt: argparse.Namespace, 
        label_mean: torch.Tensor, 
        label_std: torch.Tensor, 
        trial: optuna.trial.Trial | None = None
    ) -> tuple[float, np.ndarray]:
    """Run one training epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    num_batches = len(dataloader)

    print(f"\nEpoch: {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if outputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}")

        # Compute loss for the current batch
        batch_loss = compute_weighted_loss(outputs, targets, weights, criterion) # Compute the weighted loss across active dimensions
        orth_loss = compute_orth_loss_model(model) if opt.orth_loss_weight > 0 else 0.0 # Compute the orthogonality loss if requested
        loss = batch_loss + opt.orth_loss_weight * orth_loss
        loss.backward()

        clip_gradient(optimizer, opt.grad_clip) if opt.grad_clip > 0.0 else None
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach())
        all_targets.append(targets.detach())

        progress = (batch_idx + 1) / num_batches 
        bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
        print(f"Training: |{bar}| {progress * 100:.2f}% [{batch_idx + 1}/{num_batches}]", end="\r")

    print(" " * 100, end="\r")  # Clear the progress bar line
    avg_loss = total_loss / max(num_batches, 1)

    # Calculate unnormalized RMSE per dimension across the entire epoch
    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    rmse_per_dim = compute_unnormlized_rmse(preds_cat, targets_cat, label_mean, label_std)

    return avg_loss, rmse_per_dim


def run_training(
        opt: argparse.Namespace, 
        trial: optuna.trial.Trial | None = None
    ) -> float:
    set_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and opt.device == "cuda" else "cpu")

    print(f"Starting training {opt.model} on {opt.dataset} dataset")
    print(f"Seed set to: {opt.seed}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() and opt.device == 'cuda' else 'cpu'}")
    print(f"Training for {opt.epochs} epochs with batch size {opt.batch_size}")
    print(f"Learning rate: {opt.learning_rate}, Weight decay: {opt.weight_decay}, Optimizer: {opt.optimizer}")
    print(f"Dropout rate: {opt.dropout_rate}, Gradient clipping: {opt.grad_clip}, Data augmentation: {opt.data_augmentation}")
    print(f"Loss weights - Valence: {opt.weights_VAD[0]}, Arousal: {opt.weights_VAD[1]}, Dominance: {opt.weights_VAD[2]}, Orthogonality Loss Weight: {opt.orth_loss_weight}")
    print(f"Scheduler parameters - Factor: {opt.lr_factor}, Patience: {opt.lr_patience}, Threshold: {opt.lr_threshold}, Cooldown: {opt.lr_cooldown}, Min LR: {opt.lr_min}")

    input_size = 112

    # Determine which dimensions to include based on the provided weights
    target_names = []
    active_weights = []
    for name, w in zip(["Valence", "Arousal", "Dominance"], opt.weights_VAD):
        if w > 0:
            target_names.append(name)
            active_weights.append(w)

    if not active_weights:
        raise ValueError("At least one of weight_V, weight_A, or weight_D must be greater than 0.")

    weights_tensor = torch.tensor(active_weights, dtype=torch.float32, device=device)
    num_outputs = len(active_weights)

    include_V, include_A, include_D = opt.weights_VAD[0] > 0, opt.weights_VAD[1] > 0, opt.weights_VAD[2] > 0

    # Build folder name based on hyperparameters
    aug_str = "_data-aug" if opt.data_augmentation else ""
    folder_name = (
        f"{opt.model}/seed{opt.seed}_dataset-{opt.dataset}_V{opt.weights_VAD[0]:.1f}_A{opt.weights_VAD[1]:.1f}_D{opt.weights_VAD[2]:.1f}_orthloss{opt.orth_loss_weight:.1f}"
        f"_opt-{opt.optimizer}_lr{opt.learning_rate:.5f}_bs{opt.batch_size}_dropout{opt.dropout_rate:.1f}{aug_str}"
        f"_lr-factor{opt.lr_factor:.1f}_lr-patience{opt.lr_patience}"
    )
    path = os.path.join(opt.output_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    print(f"Output directory: {path}")

    # Load the model and optionally load pretrained weights
    model = load_model(opt.model, num_channels=3, num_outputs=num_outputs, dropout_rate=opt.dropout_rate, freezed=opt.freezed)
    pretrain_dataset_name = None
    if opt.pretrained:
        model, pretrain_dataset_name = load_pretrained_weights(model, opt.model)
        print(f"Loaded pretrained weights for {opt.model} trained on {pretrain_dataset_name} dataset.")

    model.to(device)

    # Target label statistics and image statistics
    DataLoader._ensure_image_stats(opt.dataset)
    DataLoader._ensure_label_stats(opt.dataset)

    image_mean, image_std = (DataLoader.image_mean, DataLoader.image_std) if not pretrain_dataset_name == "ms1m" else (np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
    image_mean, image_std = torch.tensor(image_mean, dtype=torch.float32, device=device), torch.tensor(image_std, dtype=torch.float32, device=device)
    label_mean, label_std = DataLoader.label_mean, DataLoader.label_std
    label_mean, label_std = torch.tensor(label_mean, dtype=torch.float32, device=device), torch.tensor(label_std, dtype=torch.float32, device=device)

    # Transforms for training and validation datasets
    if opt.data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])

    # Load datasets and create dataloaders
    DataLoader.set_data_protocol("small_split")
    train_dataset = DataLoader(dataset=opt.dataset, split="Train", include_V=include_V, include_A=include_A, include_D=include_D, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    val_dataset = DataLoader(dataset=opt.dataset, split="Val", include_V=include_V, include_A=include_A, include_D=include_D, transform=val_transform)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # Loss and optimizer setup
    criterion = torch.nn.MSELoss(reduction='none')

    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt.optimizer}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=opt.lr_factor, patience=opt.lr_patience,
        threshold=opt.lr_threshold, threshold_mode=opt.lr_threshold_mode,
        cooldown=opt.lr_cooldown, min_lr=opt.lr_min,
    )

    # Resuming from checkpoint if available
    checkpoint_path = os.path.join(path, "checkpoint.pth")
    start_epoch = 0
    best_score = float("inf")
    early_stop_counter = 0

    if opt.resume and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = float(checkpoint.get("best_score", float("inf")))
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))

    # Logging setup
    log_file = os.path.join(path, "log.csv")
    write_header = not (opt.resume and os.path.exists(log_file) and os.path.getsize(log_file) > 0)
    header = ["epoch", "train_loss"] + [f"{name}_train_rmse" for name in target_names] + ["val_loss"] + [f"{name}_val_rmse" for name in target_names]

    if write_header:
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Training loop
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(start_epoch, opt.epochs):
        train_loss, train_rmse_per_dim = train(epoch, trainloader, model, optimizer, criterion, weights_tensor, device, opt, label_mean, label_std, trial)
        val_loss, val_rmse_per_dim = evaluate(valloader, model, criterion, weights_tensor, label_mean = label_mean, label_std = label_std, device=device, trial=trial)

        # Print epoch summary
        train_str = ", ".join([f"{name} RMSE: {val:.4f}" for name, val in zip(target_names, train_rmse_per_dim)])
        val_str = ", ".join([f"{name} RMSE: {val:.4f}" for name, val in zip(target_names, val_rmse_per_dim)])
        print(f"Train Loss: {train_loss:.4f} | {train_str}\nVal Loss: {val_loss:.4f} | {val_str}")

        # Update learning rate scheduler and check for early stopping
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != prev_lr:
            print("Learning rate reduced: {:.6f} -> {:.6f}".format(prev_lr, current_lr))
            prev_lr = current_lr

        # Save epoch results to log file
        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f)
            row = [epoch + 1, train_loss] + train_rmse_per_dim.tolist() + [val_loss] + val_rmse_per_dim.tolist()
            writer.writerow(row)

        # Check for improvement and save the best model
        if val_loss < best_score:
            best_score = val_loss
            early_stop_counter = 0
            if not opt.no_model_save:
                print("Saving best model...")
                torch.save(model.state_dict(), os.path.join(path, "best_model_state.pth"))
        else:
            early_stop_counter += 1

        if not opt.no_checkpoint:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "early_stop_counter": early_stop_counter,
            }
            save_checkpoint(checkpoint, os.path.join(path, "checkpoint.pth"))

        if early_stop_counter >= opt.early_stopping_patience:
            print("Early stopping triggered. No improvement in validation loss for {} epochs.".format(opt.early_stopping_patience))
            break

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    return best_score


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="fer", choices=["fer", "caers", "afew"], help="Dataset name (default: fer)")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the optimizer (default: 5e-4)")
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "efficientnet", "mobilenet", "mobilefacenet"], help="Model architecture to use (default: vgg16)")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights for the model")
    parser.add_argument("--freezed", action="store_true", help="Freeze the convolutional layers of the model")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the regression head (default: 0.5)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "adamw"], help="Optimizer to use (default: adam)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training (default: cuda if available, otherwise cpu)")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping value (default: 0.0, no clipping)")
    parser.add_argument("--data_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint if available")
    parser.add_argument("--no_checkpoint", action="store_true", help="Disable checkpoint saving")
    parser.add_argument("--no_model_save", action="store_true", help="Disable model saving")
    parser.add_argument("--weights_VAD", type=parse_weights_vad, default=[1.0, 1.0, 1.0], help="Weights for the V, A, and D losses (default: 1.0,1.0,1.0)")
    parser.add_argument("--orth_loss_weight", type=float, default=0.5, help="Weight for the orthogonality loss (default: 0.5)")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="Factor by which to reduce learning rate (default: 0.1)")
    parser.add_argument("--lr_patience", type=int, default=10, help="Number of epochs to wait for improvement before reducing learning rate (default: 10)")
    parser.add_argument("--lr_threshold", type=float, default=1e-4, help="Minimum change in loss to qualify as improvement (default: 1e-4)")
    parser.add_argument("--lr_threshold_mode", type=str, default="rel", help="Mode to use for determining if loss has improved (default: rel)")
    parser.add_argument("--lr_cooldown", type=int, default=0, help="Number of epochs to wait before resuming normal operation after reducing learning rate (default: 0)")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum learning rate (default: 0.0)")

    return parser


def main():
    parser = build_parser()
    opt = parser.parse_args()
    run_training(opt)


if __name__ == "__main__":
    main()
