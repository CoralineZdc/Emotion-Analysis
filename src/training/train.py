"""Training execution module featuring an encapsulated Trainer class."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import torch

from src.evaluation.test import evaluate
from src.utils import (
    VADDataset,
    VADLoss,
    compute_metrics,
    get_parameter_groups,
    get_transforms,
    load_model,
    load_pretrained_weights,
    parse_csv_floats,
    set_backbone_trainable,
    set_seed,
)


class Trainer:
    """Encapsulates execution state, optimization loop, and metric logging."""

    def __init__(
        self, opt: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None
    ) -> None:
        self.opt = opt
        self.trial = trial
        self.is_optuna = trial is not None
        self.display = not self.is_optuna
        self.device = torch.device("cuda" if torch.cuda.is_available() and opt.device == "cuda" else "cpu")
        self.num_channels = 1 if opt.pretrained and opt.weights_source == "custom" else 3

        set_seed(opt.seed)
        self._setup_target_weights()
        if self.display:
            self._setup_logging()
        self._setup_dataloaders()
        self._setup_model()
        self._setup_optimization()


    def _setup_target_weights(self) -> None:
        """Sets up the active target names and weights based on the provided VAD weights and dataset."""
        if self.opt.dataset.lower() == "afew":
            self.opt.VAD_weights = [self.opt.VAD_weights[0], self.opt.VAD_weights[1], 0.0]

        self.target_names, active_weights = [], []
        for name, w in zip(["Valence", "Arousal", "Dominance"], self.opt.VAD_weights):
            if w > 0:
                self.target_names.append(name)
                active_weights.append(w)

        if not active_weights:
            raise ValueError("At least one VAD weight must be positive.")

        self.weights_tensor = torch.tensor(active_weights, dtype=torch.float32, device=self.device)


    def _setup_dataloaders(self) -> None:
        """Sets up the training and validation dataloaders based on the dataset and transformations."""
        image_mean  = [0.485, 0.456, 0.406] if self.opt.weights_source == "imagenet" else [0.5] * self.num_channels
        image_std   = [0.229, 0.224, 0.225] if self.opt.weights_source == "imagenet" else [0.5] * self.num_channels
        train_tf, val_tf = get_transforms(image_mean, image_std, self.opt.input_size, self.opt.data_augmentation)

        v_inc, a_inc, d_inc = [w > 0 for w in self.opt.VAD_weights[:3]]

        train_ds = VADDataset(self.opt.dataset, "Train", v_inc, a_inc, d_inc, self.num_channels, train_tf, display=self.display)
        val_ds = VADDataset(self.opt.dataset, "Val", v_inc, a_inc, d_inc, self.num_channels, val_tf, display=self.display)

        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers)
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers)


    def _setup_model(self) -> None:
        """Instantiates the model architecture, loads pretrained weights if specified, and moves the model to the appropriate device."""
        if self.opt.resume:
            state_dict_path = Path(self.opt.output_dir) / self._setup_folder_name() / "best_model_state.pth"
            if state_dict_path.exists():
                print(f"Resuming training from checkpoint: {state_dict_path}")
                self.model = torch.load(state_dict_path, map_location=self.device)
                return
            else:
                print(f"Checkpoint not found at {state_dict_path}. Cannot resume training.")

        self.model = load_model(
            model_name=self.opt.model, 
            num_channels=self.num_channels, 
            num_outputs=len(self.weights_tensor), 
            dropout_rate=self.opt.dropout_rate, 
            freezed=self.opt.freezed, 
            display=self.display
        )
        if self.opt.pretrained:
            self.model, _ = load_pretrained_weights(
                model=self.model, 
                model_name=self.opt.model, 
                weights_source=self.opt.weights_source, 
                display=self.display
                )

        self.model.to(self.device)


    def _setup_optimization(self) -> None:
        """Sets up the optimizer, loss criterion, and learning rate scheduler based on the provided options."""
        groups = get_parameter_groups(self.model, self.opt.head_lr, self.opt.backbone_lr, self.opt.weight_decay)
        opts = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        self.optimizer = opts[self.opt.optimizer.lower()](groups)

        self.criterion = VADLoss(self.opt.criterion, weights=self.weights_tensor, alpha=self.opt.ccc_weight)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.opt.use_amp and self.device.type == "cuda")

        if self.opt.scheduler == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.opt.lr_factor, patience=self.opt.lr_patience)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.epochs)


    def _setup_folder_name(self) -> str:
        """Constructs a descriptive folder name based on the training configuration."""
        ccc_str = f"_cccweight-{self.opt.ccc_weight:.2f}" if self.opt.criterion == "combined" else ""
        freeze_str = f"_unfreeze-{self.opt.unfreeze_epoch}" if self.opt.freezed else "_freezed-False"
        folder_name = (
            f"{self.opt.model}/"
            f"seed-{self.opt.seed}_"
            f"dataset-{self.opt.dataset}_"
            f"size-{self.opt.input_size}_"
            f"weights-{self.opt.weights_source}_"
            f"{freeze_str.lstrip('_')}_"
            f"criterion-{self.opt.criterion}_"
            f"{ccc_str}_"
            f"V-{self.opt.VAD_weights[0]:.2f}_"
            f"A-{self.opt.VAD_weights[1]:.2f}_"
            f"D-{self.opt.VAD_weights[2]:.2f}_"
            f"opt-{self.opt.optimizer}_"
            f"headlr-{self.opt.head_lr:.5f}_"
            f"backbonelr-{self.opt.backbone_lr:.2f}_"
            f"decay-{self.opt.weight_decay:.1e}_"
            f"bs-{self.opt.batch_size}_"
            f"dropout-{self.opt.dropout_rate:.2f}_"
            f"aug-{self.opt.data_augmentation}_"
            f"scheduler-{''.join(self.opt.scheduler.split('_'))}_"
        )
        return folder_name


    def _setup_logging(self) -> None:
        """Sets up the logging directory and initializes the CSV log file with headers."""
        folder_name = self._setup_folder_name()
        self.log_path = Path(self.opt.output_dir) / folder_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        print(f"\nLogs and checkpoints will be saved to: {self.log_path}")

        log_file = self.log_path / "log.csv"

        if log_file.exists() and self.opt.resume:
            print(f"Resuming logging to existing file: {log_file}")
            return
        
        header = [
            "epoch", "train_loss", "train_mse", "train_rmse", "train_ccc",
            "val_loss", "val_mse", "val_rmse", "val_ccc"
        ]
        for metric in ["mse", "rmse", "ccc"]:
            header.extend([f"train_{metric}_{n}" for n in self.target_names])
            header.extend([f"val_{metric}_{n}" for n in self.target_names])

        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(header)


    def _log_metrics(
            self, 
            epoch: int, 
            train_loss: float, 
            train_m: Dict[str, Union[float, List[float]]], 
            val_loss: float, 
            val_m: Dict[str, Union[float, List[float]]]
        ) -> None:
        """Logs the metrics for the current epoch to the CSV log file."""
        log_file = self.log_path / "log.csv"
        row = [
            epoch, train_loss, train_m["mse_overall"], train_m["rmse_overall"], train_m["ccc_overall"],
            val_loss, val_m["mse_overall"], val_m["rmse_overall"], val_m["ccc_overall"],
        ]
        row.extend(train_m["mse_per_dim"] + val_m["mse_per_dim"])
        row.extend(train_m["rmse_per_dim"] + val_m["rmse_per_dim"])
        row.extend(train_m["ccc_per_dim"] + val_m["ccc_per_dim"])

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow(row)


    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, Union[float, List[float]]]]:
        """Executes a single training epoch and returns the average loss and computed metrics (mse, rmse, ccc and std)."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []
        num_batches = len(self.train_loader)

        if self.display:
            head_lr = self.optimizer.param_groups[0]["lr"]
            backbone_lr = self.optimizer.param_groups[1]["lr"] if len(self.optimizer.param_groups) > 1 else head_lr
            print(f"\nEpoch: {epoch} | Head LR: {head_lr:.6f} | Backbone LR: {backbone_lr:.6f}")

        for batch_id, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Use automatic mixed precision (AMP) for forward pass and loss computation if enabled
            with torch.amp.autocast("cuda", enabled=self.opt.use_amp and self.device.type == "cuda"):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward() # Scale the loss for AMP and perform backpropagation

            # Gradient clipping if max_grad_norm is set
            if self.opt.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.max_grad_norm)

            # Update the optimizer and scaler for AMP
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

            if self.display:
                progress = (batch_id + 1) / num_batches
                bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
                print(f"Training: |{bar}| {progress * 100:.2f}% [{batch_id + 1}/{num_batches}]", end="\r")

        if self.display:
            print(" " * 80, end="\r")

        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_metrics(torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0))
        return avg_loss, metrics



    def run(self) -> Tuple[float, Dict[str, Union[float, List[float]]], List[str]]:
        """Runs the full training loop, including evaluation and early stopping, and returns the best validation loss, metrics, and target names."""
        if self.display:
            print(f"\nStarting training {self.opt.model} on {self.opt.dataset} dataset ({len(self.weights_tensor)} outputs)")
            print(f"Using device: {self.device} | Batch Size: {self.opt.batch_size} | Epochs: {self.opt.epochs}")

        best_val_loss = float("inf")
        best_metrics = {f"{metric}_overall": "N/A" for metric in ["mse", "rmse", "ccc"]} | {f"{metric}_per_dim_{dim}": "N/A" for metric in ["mse", "rmse", "ccc"] for dim in self.target_names} | {f"pred_std_{dim}": "N/A" for dim in self.target_names}
        early_stop_counter = 0

        for epoch in range(self.opt.epochs):
            # Handle mid-training unfreezing of the backbone if specified
            if self.opt.freezed and self.opt.unfreeze_epoch >= 0 and epoch == self.opt.unfreeze_epoch:
                if self.display:
                    print(f"\n>>> Epoch {epoch}: Unfreezing backbone layers for fine-tuning <<<")
                set_backbone_trainable(self.model, trainable=True)

            # Train for one epoch and evaluate on the validation set
            train_loss, train_m = self.train_epoch(epoch)
            val_loss, val_m = evaluate(self.val_loader, self.model, criterion=self.criterion, device=self.device, display=self.display)
            val_loss = float(val_loss.item() if hasattr(val_loss, "item") else val_loss)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Update best validation loss and metrics, handle early stopping, and save the best model state if applicable
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_m
                early_stop_counter = 0
                if not self.opt.no_model_save:
                    torch.save(self.model.state_dict(), self.log_path / "best_model_state.pth")
            else:
                early_stop_counter += 1

            if self.display:
                print(f"Train Loss: {train_loss:.4f} | MSE: {train_m['mse_overall']:.4f} | RMSE: {train_m['rmse_overall']:.4f} | CCC: {train_m['ccc_overall']:.4f}")
                print(f"Val Loss:   {val_loss:.4f} | MSE: {val_m['mse_overall']:.4f} | RMSE: {val_m['rmse_overall']:.4f} | CCC: {val_m['ccc_overall']:.4f}")
                std_str = " | ".join(f"{n}: {std:.4f}" for n, std in zip(self.target_names, train_m['pred_std']))
                rmse_str = " | ".join(f"{n}: {rmse:.4f}" for n, rmse in zip(self.target_names, val_m['rmse_per_dim']))
                ccc_str = " | ".join(f"{n}: {ccc:.4f}" for n, ccc in zip(self.target_names, val_m['ccc_per_dim']))
                print(f"Pred StdDev      -> {std_str}")
                print(f"Per-Dim Val RMSE -> {rmse_str}")
                print(f"Per-Dim Val CCC  -> {ccc_str}")
                self._log_metrics(epoch, train_loss, train_m, val_loss, val_m)
            else:
                print(f"[Trial {self.trial.number} | Epoch {epoch} ] Train Loss: {train_loss:6.4f} -> RMSE: {train_m['rmse_overall']:6.4f}, CCC: {train_m['ccc_overall']:6.4f}"
                      f" | Val Loss: {val_loss:6.4f} -> RMSE: {val_m['rmse_overall']:6.4f}, CCC: {val_m['ccc_overall']:6.4f}", end="\r")


            if early_stop_counter >= self.opt.early_stopping_patience:
                if self.display:
                    print(f"\n>>> Early stopping at epoch {epoch} <<<")
                break

            # Report intermediate results to Optuna for pruning if applicable
            if self.is_optuna:
                self.trial.report(val_loss, step=epoch)
                self.trial.set_user_attr("best_metrics", best_metrics)
                if self.trial.should_prune():
                    print(" " * 150, end="\r")
                    raise optuna.exceptions.TrialPruned()

        return best_val_loss, best_metrics, self.target_names



def run_training(opt: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None):
    """Initializes the Trainer and starts the training process."""
    trainer = Trainer(opt, trial)
    return trainer.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save logs and checkpoints (default: outputs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dataset", type=str, default="fer", choices=["fer", "caers", "afew", "emotic", "emotic-child", "heco"], help="Dataset to use for training and evaluation (default: fer)")
    parser.add_argument("--input_size", type=int, default=112, help="Image spatial resolution (default: 112)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader subprocess workers (default: 4)")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Number of epochs with no improvement after which training will be stopped (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Learning rate for the optimizer head (default: 1e-4)")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="Learning rate for the optimizer backbone (default: 1.0)")
    parser.add_argument("--unfreeze_epoch", type=int, default=0, help="Epoch at which to unfreeze frozen backbone (-1 disables mid-training unfreeze)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the optimizer (default: 5e-4)")
    parser.add_argument("--model", type=str, default="vgg16", choices=["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "efficientnet", "mobilenet", "mobilefacenet"], help="Model architecture to use (default: vgg16)")
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained weights (default: False)")
    parser.add_argument("--weights_source", type=str, default="imagenet", choices=["imagenet", "custom"], help="Source of pre-trained weights (default: imagenet)")
    parser.add_argument("--freezed", action="store_true", help="Freeze the convolutional layers of the model (default: False)")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the regression head (default: 0.5)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"], help="Optimizer to use for training (default: sgd)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum norm for gradient clipping (default: 1.0, set 0 to disable)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device to use for training (default: cuda if available, otherwise cpu)")
    parser.add_argument("--data_augmentation", action="store_true", help="Apply data augmentation during training (default: False)")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP) training (default: True)")
    parser.add_argument("--resume", action="store_true", help="Resume training from a previous checkpoint (default: False)")
    parser.add_argument("--no_model_save", action="store_true", help="Do not save the model (default: False)")
    parser.add_argument("--criterion", type=str, default="mse", choices=["mse", "ccc", "combined"], help="Loss function to use for training (default: mse)")
    parser.add_argument("--VAD_weights", type=parse_csv_floats, default="1.0, 1.0, 1.0", help="Weights for the VAD loss (default: \"1.0, 1.0, 1.0\")")
    parser.add_argument("--ccc_weight", type=float, default=0.5, help="Weight for the CCC loss (default: 0.5)")
    parser.add_argument("--scheduler", type=str, default="reduce_on_plateau", choices=["reduce_on_plateau", "cosine_annealing"], help="Learning rate scheduler to use (default: reduce_on_plateau)")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced (default: 0.1)")
    parser.add_argument("--lr_patience", type=int, default=10, help="Number of epochs with no improvement after which the learning rate will be reduced (default: 10)")
    return parser


if __name__ == "__main__":
    opt = build_parser().parse_args()
    run_training(opt)