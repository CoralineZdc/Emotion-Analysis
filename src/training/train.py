import argparse
import csv
import os
import numpy as np
import optuna
import torch
from torchvision.transforms import v2 as transforms

from src.evaluation.test import evaluate
from src.utils.data_loader import DataLoader
from src.utils.training_utils import *
from src.utils.parsing_utils import parse_csv_floats

def save_checkpoint(state: dict, filename: str) -> None:
    """Save a training checkpoint to disk."""
    torch.save(state, filename)


def get_parameter_groups(
    model: torch.nn.Module, 
    base_lr: float, 
    backbone_lr_scale: float, 
    weight_decay: float
) -> list[dict]:
    if hasattr(model, "head") and hasattr(model, "backbone"):
        head_params = list(model.head.parameters())
        backbone_params = list(model.backbone.parameters())
    else:
        head_params = [p for n, p in model.named_parameters() if "head" in n.lower()]
        backbone_params = [p for n, p in model.named_parameters() if "head" not in n.lower()]

    return [
        {"params": head_params, "lr": base_lr, "weight_decay": weight_decay, "name": "head"},
        {"params": backbone_params, "lr": base_lr * backbone_lr_scale, "weight_decay": weight_decay, "name": "backbone"}
    ]


def train(
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    device: torch.device,
    opt: argparse.Namespace,
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
        head_lr = optimizer.param_groups[0]["lr"]
        backbone_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else head_lr
        print(f"\nEpoch: {epoch} | Head LR: {head_lr:.6f} | Backbone LR: {backbone_lr:.6f}")

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if opt.data_augmentation and model.training:
            inputs, targets = apply_mixup(inputs, targets, alpha=0.2)

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

        total_loss += batch_loss.item() # Only register main loss for logging
        all_preds.append(outputs.detach().cpu()) 
        all_targets.append(targets.detach().cpu())

        if not is_optuna:
            progress = (batch_idx + 1) / num_batches
            bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
            print(f"Training: |{bar}| {progress * 100:.2f}% [{batch_idx + 1}/{num_batches}]", end="\r")

    if not is_optuna:
        print(" " * 100, end="\r")

    avg_loss = total_loss / max(num_batches, 1)

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    rmse_per_dim = torch.sqrt(torch.mean((preds_cat - targets_cat) ** 2, dim=0)).cpu().numpy()

    return avg_loss, rmse_per_dim


def run_training(opt: argparse.Namespace, trial: optuna.trial.Trial | None = None) -> tuple[float, np.ndarray, list[str]]:
    set_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and opt.device == "cuda" else "cpu")
    is_optuna = trial is not None

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

        aug_str = "_dataaug" if opt.data_augmentation else ""
        ccc_str = f"_CCCweight{opt.ccc_weight:.2f}" if opt.criterion == "combined" else ""
        folder_name = (
            f"{opt.model}/seed{opt.seed}_dataset-{opt.dataset}{aug_str}_modelweights-{opt.weights_source}_criterion-{opt.criterion}{ccc_str}"
            f"_V{opt.VAD_weights[0]:.1f}_A{opt.VAD_weights[1]:.1f}_D{opt.VAD_weights[2]:.1f}"
            f"_opt-{opt.optimizer}_lr{opt.learning_rate:.5f}_backboneLRscale{opt.backbone_lr_scale:.2f}"
            f"_bs{opt.batch_size}_dropout{opt.dropout_rate:.1f}"
        )
        path = os.path.join(opt.output_dir, folder_name)
        os.makedirs(path, exist_ok=True)
        print(f"Logs and checkpoints will be saved to: {path}")

    # Model Initialization
    num_channels = 1 if opt.pretrained and opt.weights_source == "custom" else 3
    model = load_model(opt.model, num_channels=num_channels, num_outputs=num_outputs, dropout_rate=opt.dropout_rate, freezed=opt.freezed, display=not is_optuna)
    pretrain_dataset_name = None
    if opt.pretrained:
        model, pretrain_dataset_name = load_pretrained_weights(
            model=model, 
            model_name=opt.model, 
            weights_source=opt.weights_source,
            display=not is_optuna
        )

    model.to(device)

    # Target label statistics and image statistics
    DataLoader.set_data_protocol("small_split")
    DataLoader.set_num_channels(num_channels)
    DataLoader._ensure_image_stats(opt.dataset)
    DataLoader._ensure_label_stats(opt.dataset)

    if pretrain_dataset_name == "ms1m":
        image_mean = [0.5]if num_channels == 1 else [0.5, 0.5, 0.5]
        image_std = [0.5] if num_channels == 1 else [0.5, 0.5, 0.5]
    elif pretrain_dataset_name == "imagenet":
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
    else:
        image_mean = DataLoader.image_mean
        image_std = DataLoader.image_std
        if isinstance(image_mean, (torch.Tensor, np.ndarray)):
            image_mean = image_mean.tolist()
        if isinstance(image_std, (torch.Tensor, np.ndarray)):
            image_std = image_std.tolist()
        if num_channels == 1 and len(image_mean) > 1:
            image_mean = [image_mean[0]]
            image_std = [image_std[0]]

    # Transforms

    padding_size = int(opt.input_size * 0.15)
    crop_resolution = opt.input_size
    
    if opt.data_augmentation:
        train_transform = transforms.Compose([
            # 1. Resize & Random Crop
            transforms.Resize((opt.input_size + padding_size, opt.input_size + padding_size)),
            transforms.RandomCrop((crop_resolution, crop_resolution)),
            
            # 2. Geometric Transformations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-10.0, 10.0)),
            
            # 3. Photometric Augmentations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            
            # 4. Conversion & Normalization
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),  # Replaces ToTensor (converts [0, 255] uint8 -> [0.0, 1.0] float32)
            transforms.Normalize(mean=image_mean, std=image_std),
            
            # 5. Facial Occlusions
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((opt.input_size, opt.input_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((opt.input_size, opt.input_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    # DataLoaders
    train_dataset = DataLoader(opt.dataset, split="Train", include_V=include_V, include_A=include_A, include_D=include_D, transform=train_transform, display=not is_optuna)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    val_dataset = DataLoader(opt.dataset, split="Val", include_V=include_V, include_A=include_A, include_D=include_D, transform=val_transform, display=not is_optuna)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # Optimizer with Parameter Groups for Differential Learning Rates
    param_groups = get_parameter_groups(
        model, 
        base_lr=opt.learning_rate, 
        backbone_lr_scale=opt.backbone_lr_scale, 
        weight_decay=opt.weight_decay
    )

    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
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
    if not is_optuna:
        log_file = os.path.join(path, "log.csv")
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss"] + [f"{n}_train_rmse" for n in target_names] + ["val_loss"] + [f"{n}_val_rmse" for n in target_names])

    best_score = float("inf")
    best_val_rmse = None
    early_stop_counter = 0

    for epoch in range(opt.epochs):
        # Mid-training unfreezing check
        if opt.freezed and opt.unfreeze_epoch >= 0 and epoch == opt.unfreeze_epoch:
            if not is_optuna:
                print(f"\n>>> Epoch {epoch}: Unfreezing backbone layers for fine-tuning <<<")
            for name, param in model.named_parameters():
                param.requires_grad = True

        train_loss, train_rmse = train(epoch, trainloader, model, optimizer, weights_tensor, device, opt, trial)
        val_loss, val_rmse = evaluate(valloader, model, opt.criterion, alpha=opt.ccc_weight, weights=weights_tensor, device=device, trial=trial)
        val_loss_float = float(val_loss.item()) if hasattr(val_loss, "item") else float(val_loss)

        if not is_optuna:
            print(f"Train Loss: {train_loss:.4f} | {' | '.join(f'Train RMSE {name}: {rmse:.4f}' for name, rmse in zip(target_names, train_rmse))}")
            print(f"Val Loss: {val_loss:.4f} | {' | '.join(f'Val RMSE {name}: {rmse:.4f}' for name, rmse in zip(target_names, val_rmse))}")

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss] + train_rmse.tolist() + [val_loss_float] + val_rmse.tolist())

        scheduler.step(val_loss_float)

        if val_loss_float < best_score:
            best_score = val_loss_float
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
    parser.add_argument("--dataset", type=str, default="fer", choices=["fer", "caers", "afew", "emotic", "emotic-child", "heco"], help="Dataset to use for training and evaluation (default: fer)")
    parser.add_argument("--input_size", type=int, default=112, help="Image spatial resolution (default: 112)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader subprocess workers (default: 4)")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of epochs with no improvement after which training will be stopped (default: 20)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save logs and model checkpoints (default: ./output)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer head (default: 1e-4)")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1, help="Scale factor for backbone learning rate relative to head LR (default: 0.1)")
    parser.add_argument("--unfreeze_epoch", type=int, default=0, help="Epoch at which to unfreeze frozen backbone (-1 disables mid-training unfreeze)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the optimizer (default: 5e-4)")
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "efficientnet", "mobilenet", "mobilefacenet"], help="Model architecture to use (default: vgg16)")
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained weights (default: False)")
    parser.add_argument("--weights_source", type=str, default="imagenet", choices=["imagenet", "custom"], help="Source of pre-trained weights (default: imagenet)")
    parser.add_argument("--freezed", action="store_true", help="Freeze the convolutional layers of the model (default: False)")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the regression head (default: 0.5)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"], help="Optimizer to use for training (default: sgd)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device to use for training (default: cuda if available, otherwise cpu)")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping value (default: 0.0, no clipping)")
    parser.add_argument("--data_augmentation", action="store_true", help="Apply data augmentation during training (default: False)")
    parser.add_argument("--resume", action="store_true", help="Resume training from a previous checkpoint (default: False)")
    parser.add_argument("--no_checkpoint", action="store_true", help="Do not save checkpoints (default: False)")
    parser.add_argument("--no_model_save", action="store_true", help="Do not save the model (default: False)")
    parser.add_argument("--criterion", type=str, default="mse", choices=["mse", "ccc", "combined"], help="Loss function to use for training (default: mse)")
    parser.add_argument("--VAD_weights", type=parse_csv_floats, default="1.0, 1.0, 1.0", help="Weights for the VAD loss (default: \"1.0, 1.0, 1.0\")")
    parser.add_argument("--orth_loss_weight", type=float, default=0.0, help="Weight for the orthogonal loss (default: 0.5)")
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