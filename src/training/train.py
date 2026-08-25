import argparse
import os
import random
import numpy as np
import torch
import csv
import optuna

from src.utils.training_utils import *
from src.utils.data_loader import DataLoader
import src.transforms.transforms as transforms


def save_checkpoint(state, filename):
    """Save a training checkpoint to disk."""
    torch.save(state, filename)


def train(epoch, dataloader, net, optimizer, loss, use_cuda, opt, trial=None):
    """Run one training epoch and return the average loss."""
    net.train()
    total_loss = 0.0
    total_batches = 0
    device = torch.device("cuda" if use_cuda else "cpu")

    weights = torch.tensor([opt.weight_V, opt.weight_A, opt.weight_D], device=device)
    if weights.sum() == 0:
        raise ValueError("All weights for V, A, D are zero. At least one must be non-zero.")

    all_predictions = []
    all_targets = []

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

        loss_fct = torch.nn.MSELoss(reduction='none') if opt.loss == "mse" else torch.nn.L1Loss(reduction='none')
        loss_per_dim = loss_fct(outputs, targets)  # Use the specified loss function

        weighted_loss_per_dim = loss_per_dim * weights / weights.sum()  # Normalize weights to sum to 1

        batch_loss = weighted_loss_per_dim.sum(dim=1).mean()  # Average the weighted loss across dimensions
        orth_loss = compute_orth_loss_model(net)
        loss = batch_loss + opt.orth_loss_weight * orth_loss

        all_predictions.append(outputs.detach())
        all_targets.append(targets.detach())

        loss.backward()

        if opt.grad_clip > 0.0:
            clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    preds_cat = torch.cat(all_predictions, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    #preds_raw = preds_cat * label_std + label_mean #Uncomment this line if you want to denormalize the predictions
    #targets_raw = targets_cat * label_std + label_mean #Uncomment this line if you want to denormalize the targets

    rmse_per_dim = torch.sqrt(torch.mean((preds_cat - targets_cat) ** 2, dim=0)).cpu().numpy()


    avg_loss = total_loss / max(total_batches, 1)
    print("Train Loss: {:.4f}                                   ".format(avg_loss))
    print(f"   Valence RMSE: {rmse_per_dim[0]:.4f}, Arousal RMSE: {rmse_per_dim[1]:.4f}, Dominance RMSE: {rmse_per_dim[2]:.4f}")
    return avg_loss, rmse_per_dim


def evaluate(dataloader, model, loss, use_cuda, opt, trial=None):
    model.eval()
    device = torch.device("cuda" if use_cuda else "cpu")
    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    weights = torch.tensor([opt.weight_V, opt.weight_A, opt.weight_D], device=device)

    with torch.no_grad():
        for inputs, targets in dataloader:
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            if outputs.shape != targets.shape:
                raise ValueError(
                    "Model outputs and targets must have the same shape, "
                    f"got outputs={tuple(outputs.shape)}, targets={tuple(targets.shape)}"
                )

            loss_fct = torch.nn.MSELoss(reduction='none') if opt.loss == "mse" else torch.nn.L1Loss(reduction='none')
            loss_per_dim = loss_fct(outputs, targets)  # Use the specified loss function
            weighted_loss_per_dim = loss_per_dim * weights / weights.sum()  # Normalize weights to sum to 1
            batch_loss = weighted_loss_per_dim.sum(dim=1).mean()  # Average the weighted loss across dimensions

            total_loss += batch_loss.item()
            total_batches += 1

            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / max(total_batches, 1)
    predictions = torch.cat(all_predictions, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    rmse_per_dim = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    return avg_loss, rmse_per_dim


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

    data_augmentation = opt.data_augmentation

    folder_name = f"{opt.model}/seed{opt.seed}_dataset-{opt.dataset}_opt-{opt.optimizer}_lr{opt.learning_rate:.5f}_bs{opt.batch_size}_dropout{opt.dropout_rate:.1f}_{f'_data-aug' if data_augmentation else ''}_lr-factor{opt.lr_factor:.1f}_lr-patience{opt.lr_patience}_lr-threshold{opt.lr_threshold:.4f}_lr-threshold-mode-{opt.lr_threshold_mode}_lr-cooldown{opt.lr_cooldown}_lr-min{opt.lr_min:.1f}"
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
    normalization_mean, normalization_std = (DataLoader.image_mean.numpy(), DataLoader.image_std.numpy()) if not pretrain_dataset_name == "ms1m" else (np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))

    cut_size = 44

    if data_augmentation:
        transform_list = [
            transforms.RandomCrop(cut_size),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((input_size, input_size)),
            #transforms.GrayScale(num_output_channels=3) if opt.pretrained and opt.model != "efficientnet" else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std)
        ]

        train_transform = transforms.Compose(transform_list)
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            #transforms.GrayScale(num_output_channels=3) if opt.pretrained and opt.model != "efficientnet" else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std)
        ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        #transforms.GrayScale(num_output_channels=3) if opt.pretrained and opt.model != "efficientnet" else lambda x: x,
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std)
    ])

    train_dataset = DataLoader(dataset=opt.dataset, split="Train", transform=train_transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    val_dataset = DataLoader(dataset=opt.dataset, split="Val", transform=val_transform)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)


    print(
        "Model: {}".format(
            opt.model.upper()
        )
    )

    if use_cuda:
        model = model.cuda()

    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
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
            writer.writerow(["epoch", "train_loss", "V_train_rmse", "A_train_rmse", "D_train_rmse", "val_loss", "V_val_rmse", "A_val_rmse", "D_val_rmse"])

    for epoch in range(start_epoch, total_epoch):
        train_loss, train_rmse_per_dim = train(epoch, trainloader, model, optimizer, opt.loss, use_cuda, opt, trial)
        val_loss, val_rmse_per_dim = evaluate(valloader, model, opt.loss, use_cuda, opt, trial)

        print("Validation Loss: {:.4f}".format(val_loss))
        print(f"   Valence RMSE: {val_rmse_per_dim[0]:.4f}, Arousal RMSE: {val_rmse_per_dim[1]:.4f}, Dominance RMSE: {val_rmse_per_dim[2]:.4f}")

        metric_for_scheduler = val_loss
        scheduler.step(metric_for_scheduler)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != prev_lr:
            print("Learning rate reduced from {:.6f} to {:.6f}".format(prev_lr, current_lr))
            prev_lr = current_lr

        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_rmse_per_dim[0], train_rmse_per_dim[1], train_rmse_per_dim[2], val_loss, val_rmse_per_dim[0], val_rmse_per_dim[1], val_rmse_per_dim[2]])

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
    parser.add_argument("--dataset", type=str, default="fer", choices=["fer", "caers"], help="Dataset name (default: fer)")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "efficientnet", "mobilenet", "mobilefacenet"], help="Model architecture to use (default: vgg16)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the optimizer (default: 5e-4)")
    parser.add_argument("--loss", type=str, default="mse", choices=["rmse", "mse"], help="Loss metric to use (default: mse)")
    parser.add_argument("--weight_V", type=float, default=1.0, help="Weight for the V loss (default: 1.0)")
    parser.add_argument("--weight_A", type=float, default=1.0, help="Weight for the A loss (default: 1.0)")
    parser.add_argument("--weight_D", type=float, default=00, help="Weight for the D loss (default: 0.0)")
    parser.add_argument("--orth_loss_weight", type=float, default=0.5, help="Weight for the orthogonality loss (default: 0.5)")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights for the model")
    parser.add_argument("--freezed", action="store_true", help="Freeze the convolutional layers of the model")
    parser.add_argument("--num_outputs", type=int, default=3, help="Number of outputs for the regression head (default: 3)")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the regression head (default: 0.5)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "adamw"], help="Optimizer to use (default: adam)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for training if available")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping value (default: 0.0, no clipping)")
    parser.add_argument("--data_augmentation", action="store_true", help="Enable data augmentation")
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
