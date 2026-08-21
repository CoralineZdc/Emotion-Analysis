import argparse
import os
import random
import numpy as np
import torch
import csv
import optuna

from src.utils.training_utils import *
from src.utils.data_loader import DataLoader
from src.transforms import transforms 


def save_checkpoint(state, filename):
    """Save a training checkpoint to disk."""
    torch.save(state, filename)


def train(epoch, dataloader, net, optimizer, criterion, use_cuda, opt, trial=None):
    """Run one training epoch and return the average loss."""
    net.train()
    total_loss = 0.0
    total_batches = 0

    print("\nEpoch: {}".format(epoch))
    print("LR: {}".format(optimizer.param_groups[0]["lr"]))

    for inputs, targets in dataloader:
        print(f"Training: |{'█'*int((total_batches + 1) / len(dataloader) * 20)}{' '*int(20 - int((total_batches + 1) / len(dataloader) * 20))}| {(total_batches + 1) / len(dataloader) * 100 :.2f}% [{total_batches + 1}/{len(dataloader)}]", end="\r")
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)

        if outputs.shape != targets.shape:
            raise ValueError(
                "Model outputs and targets must have the same shape, "
                f"got outputs={tuple(outputs.shape)}, targets={tuple(targets.shape)}"
            )

        loss = torch.nn.L1Loss()(outputs, targets)  # Use L1 loss (Mean Absolute Error) for regression
        loss.backward()

        if opt.grad_clip > 0.0:
            clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    print("Train Loss: {:.4f}                                   ".format(avg_loss))
    return avg_loss


def evaluate(dataloader, model, criterion, use_cuda, opt, trial=None):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            if outputs.shape != targets.shape:
                raise ValueError(
                    "Model outputs and targets must have the same shape, "
                    f"got outputs={tuple(outputs.shape)}, targets={tuple(targets.shape)}"
                )

            loss = torch.nn.L1Loss()(outputs, targets)  # Use L1 loss
            
            total_loss += loss.item()
            total_batches += 1

            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / max(total_batches, 1)

    return avg_loss


def run_training(opt, trial=None):
    set_seed(opt.seed)

    input_size = 48

    DataLoader.set_data_protocol("small_split")
    print("[Small Split] Using current extracted CSVs in ./data")

    use_cuda = torch.cuda.is_available()
    if opt.cuda and not use_cuda:
        print("[Warning] --cuda was requested but CUDA is not available. Falling back to CPU.")

    start_epoch = 0
    total_epoch = opt.epochs
    early_stop_patience = opt.early_stopping_patience
    best_score = float("inf")
    early_stop_counter = 0

    data_augmentation = opt.data_augmentation_param > 0
    scale = opt.data_augmentation_param / 100.0
    rotation = opt.data_augmentation_param
    hor_shift = opt.data_augmentation_param / 100.0
    ver_shift = opt.data_augmentation_param / 100.0

    folder_name = f"{opt.model}/seed{opt.seed}_dataset-{opt.dataset}_opt-{opt.optimizer}_lr{opt.learning_rate:.5f}_bs{opt.batch_size}_dropout{opt.dropout_rate:.1f}{f'_scale{scale:.1f}_rot{rotation:.1f}_hor-shift{hor_shift:.1f}_ver-shift{ver_shift:.1f}' if data_augmentation else ''}_lr-factor{opt.lr_factor:.1f}_lr-patience{opt.lr_patience}_lr-threshold{opt.lr_threshold:.4f}_lr-threshold-mode-{opt.lr_threshold_mode}_lr-cooldown{opt.lr_cooldown}_lr-min{opt.lr_min:.1f}"
    print("Folder name for this run:", folder_name)
    path = os.path.join(opt.output_dir, folder_name)
    os.makedirs(path, exist_ok=True)


    num_channels = 3
    model = load_model(opt.model, num_channels=num_channels, num_outputs=opt.num_outputs, dropout_rate=opt.dropout_rate, freezed=opt.freezed)
    
    pretrain_dataset_name = None
    if opt.pretrained:
        model, pretrain_dataset_name = load_pretrained_weights(model, opt.model)
        print(f"Loaded pretrained weights for {opt.model} trained on {pretrain_dataset_name} dataset.")

    DataLoader._ensure_label_stats(opt.dataset)
    DataLoader._ensure_image_stats(opt.dataset)
    normalization_mean, normalization_std = DataLoader.image_mean.numpy(), DataLoader.image_std.numpy()

    if data_augmentation:
        transform_list = [
            transforms.GrayScale(num_output_channels=3) if opt.pretrained and opt.model != "efficientnet" else lambda x: x,
            transforms.Resize((input_size, input_size)),
            transforms.RandomResizedCrop(input_size, scale=(1-scale, 1+scale), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=rotation),
            transforms.RandomShift(shift_range=(hor_shift, ver_shift)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std)
        ]

        train_transform = transforms.Compose(transform_list)
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.GrayScale(num_output_channels=3) if opt.pretrained and opt.model != "efficientnet" else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std)
        ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
            transforms.GrayScale(num_output_channels=3) if opt.pretrained and opt.model != "efficientnet" else lambda x: x,
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std)
    ])

    train_dataset = DataLoader(dataset=opt.dataset, split="Train", transform=train_transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    val_dataset = DataLoader(dataset=opt.dataset, split="Val", transform=val_transform)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)


    print(
        "Model: {}".format(
            opt.model.upper()
        )
    )

    if use_cuda:
        model = model.cuda()

    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    elif opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate)
    else:
        raise ValueError("Unsupported optimizer: {}".format(opt.optimizer))
    
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

    prev_lr = optimizer.param_groups[0]["lr"]
    checkpoint_path = os.path.join(path, "checkpoint.pth")
    start_epoch = 0

    if opt.resume and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            print("Warning: checkpoint optimizer state is incompatible; using fresh optimizer state")
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            print("Warning: checkpoint scheduler state is incompatible; using fresh scheduler state")
        start_epoch = checkpoint["epoch"] + 1
        best_score = float(checkpoint.get("best_score", float("inf")))
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
        print("Resume state: best_score={:.6f}, early_stop_counter={}".format(best_score, early_stop_counter))
    else:
        if opt.no_checkpoint:
            print("Checkpoint disabled; starting from scratch")
        elif os.path.exists(checkpoint_path):
            print("Checkpoint found but resume is disabled; starting from scratch")
        else:
            print("No checkpoint found, starting from scratch")

    log_file = os.path.join(path, "log.csv")
    write_header = True
    if opt.resume and os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        write_header = False

    with open(log_file, "a" if not write_header else "w") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(start_epoch, total_epoch):
        train_loss = train(epoch, trainloader, model, optimizer, None, use_cuda, opt, trial)
        val_loss = evaluate(valloader, model, None, use_cuda, opt, trial)

        print("Validation Loss: {:.4f}".format(val_loss))

        metric_for_scheduler = val_loss
        scheduler.step(metric_for_scheduler)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != prev_lr:
            print("Learning rate reduced from {:.6f} to {:.6f}".format(prev_lr, current_lr))
            prev_lr = current_lr

        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])

        is_best = val_loss < best_score
        if is_best:
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

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered. No improvement in validation loss for {} epochs.".format(early_stop_patience))
            break

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    return best_score


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="fer", choices=["fer"], help="Dataset name (default: fer)")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "efficientnet", "mobilenet", "mobilefacenet"], help="Model architecture to use (default: vgg16)")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights for the model")
    parser.add_argument("--freezed", action="store_true", help="Freeze the convolutional layers of the model")
    parser.add_argument("--num_outputs", type=int, default=3, help="Number of outputs for the regression head (default: 3)")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the regression head (default: 0.5)")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "adamw"], help="Optimizer to use (default: adam)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for training if available")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping value (default: 0.0, no clipping)")
    parser.add_argument("--data_augmentation_param", type=int, default=5, help="Data augmentation parameter for rotation (degrees), scaling (percent), and shifting (percent) (default: 5)")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint if available")
    parser.add_argument("--no_checkpoint", action="store_true", help="Disable checkpoint saving")
    parser.add_argument("--no_model_save", action="store_true", help="Disable model saving")
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
