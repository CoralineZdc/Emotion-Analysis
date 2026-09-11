import argparse
import copy
import csv
import os
import time
from typing import Any, List

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.training.train import build_parser, run_training
from src.utils.parsing_utils import parse_csv_floats, parse_csv_ints, parse_csv_strings, parse_csv_bool_tuples


# -------------------------------------------------------------------------
# Helper Functions for Dynamic Hyperparameter Sampling
# -------------------------------------------------------------------------

def sample_float(trial: optuna.trial.Trial, name: str, bounds: List[float], log: bool = False) -> float:
    """Samples a float hyperparameter based on length of input range list."""
    if len(bounds) == 1:
        return bounds[0]
    elif len(bounds) == 2:
        return trial.suggest_float(name, bounds[0], bounds[1], log=log)
    elif len(bounds) == 3 and not log:
        return trial.suggest_float(name, bounds[0], bounds[1], step=bounds[2])
    else:
        raise ValueError(f"Invalid range specification for float '{name}': {bounds}. Expecting 1, 2, or 3 values.")


def sample_int(trial: optuna.trial.Trial, name: str, bounds: List[int]) -> int:
    """Samples an integer hyperparameter based on length of input range list."""
    if len(bounds) == 1:
        return bounds[0]
    elif len(bounds) == 2:
        return trial.suggest_int(name, bounds[0], bounds[1])
    elif len(bounds) == 3:
        return trial.suggest_int(name, bounds[0], bounds[1], step=bounds[2])
    else:
        raise ValueError(f"Invalid range specification for int '{name}': {bounds}. Expecting 1, 2, or 3 values.")


def sample_categorical(trial: optuna.trial.Trial, name: str, choices: List[Any]) -> Any:
    """Picks from discrete values or returns the single choice if only 1 provided."""
    if len(choices) == 1:
        return choices[0]
    return trial.suggest_categorical(name, choices)


# -------------------------------------------------------------------------
# Optuna Objective Function
# -------------------------------------------------------------------------


def objective(trial: optuna.trial.Trial, base_opt: argparse.Namespace, log_csv_path: str) -> float:
    """Optuna objective function optimized for Mean RMSE minimization."""
    opt = copy.deepcopy(base_opt)

    # ---------------------------------------------------------
    # 1. Hyperparameter Search Space
    # ---------------------------------------------------------

    # Disable disk I/O saving during optimization
    opt.no_checkpoint = True
    opt.no_model_save = True

    opt.weights_source = sample_categorical(trial, "weights_source", opt.range_weights_source)
    opt.unfreeze_epoch = sample_int(trial, "unfreeze_epoch", opt.range_unfreeze_epoch)
    opt.input_size = sample_categorical(trial, "input_size", opt.range_input_size)
    
    opt.learning_rate = sample_float(trial, "learning_rate", opt.range_learning_rate, log=True)
    opt.weight_decay = sample_float(trial, "weight_decay", opt.range_weight_decay, log=True)
    opt.backbone_lr_scale = sample_float(trial, "backbone_lr_scale", opt.range_backbone_lr_scale, log=True)
    opt.optimizer = sample_categorical(trial, "optimizer", opt.range_optimizer)
    opt.batch_size = sample_categorical(trial, "batch_size", opt.range_batch_size)
    opt.dropout_rate = sample_float(trial, "dropout_rate", opt.range_dropout_rate)

    # VAD Weight Normalization (Ensures sum equals 1.0 to prevent loss tricking)
    w_v = sample_float(trial, "weight_v", [0.0, 1.0])
    w_a = sample_float(trial, "weight_a", [0.0, 1.0])
    
    if opt.dataset.lower() == "afew":
        w_d = 0.0
        total_w = w_v + w_a
    else:
        w_d = sample_float(trial, "weight_d", [0.0, 1.0])
        total_w = w_v + w_a + w_d

    opt.VAD_weights = [round(w_v / total_w, 4), round(w_a / total_w, 4), round(w_d / total_w, 4)]

    # Loss Criterion Configuration
    opt.criterion = sample_categorical(trial, "criterion", opt.range_criterion)
    if opt.criterion == "combined":
        opt.ccc_weight = sample_float(trial, "ccc_weight", opt.range_ccc_weight)
    else:
        opt.ccc_weight = 0.0

    opt.orth_loss_weight = sample_float(trial, "orth_loss_weight", opt.range_orth_loss_weight)
    opt.lr_factor = sample_categorical(trial, "lr_factor", opt.range_lr_factor)
    opt.lr_patience = sample_int(trial, "lr_patience", opt.range_lr_patience)

    start_time = time.time()
    status = "COMPLETE"
    best_val_loss = float("inf")
    best_rmse_per_dim = None
    target_names = []
    mean_rmse = float("inf")

    # ---------------------------------------------------------
    # 2. Training Execution & Metric Extraction
    # ---------------------------------------------------------
    try:
        best_val_loss, best_rmse_per_dim, target_names = run_training(opt, trial=trial)

        if best_rmse_per_dim is not None and len(best_rmse_per_dim) > 0:
            rmse_floats = [float(r.item()) if hasattr(r, "item") else float(r) for r in best_rmse_per_dim]
            
            for name, rmse_val in zip(target_names, rmse_floats):
                trial.set_user_attr(f"val_rmse_{name}", round(rmse_val, 4))

            # Scale-invariant evaluation metric: Mean RMSE across active VAD targets
            mean_rmse = sum(rmse_floats) / len(rmse_floats)
            trial.set_user_attr("mean_rmse", round(mean_rmse, 4))

    except optuna.exceptions.TrialPruned:
        status = "PRUNED"
        raise
    except Exception as e:
        status = f"FAILED ({str(e)})"
        print(f"[Trial {trial.number}] Encountered exception: {e}")
        return float("inf")
    finally:
        elapsed_time = round(time.time() - start_time, 2)

        # ---------------------------------------------------------
        # 3. Comprehensive Logging to CSV
        # ---------------------------------------------------------
        file_exists = os.path.exists(log_csv_path)

        fixed_param_keys = [
            "weights_source", "unfreeze_epoch", "input_size", "learning_rate", 
            "weight_decay", "backbone_lr_scale", "optimizer", "batch_size", 
            "dropout_rate", "weight_v", "weight_a", "weight_d", "criterion", 
            "ccc_weight", "orth_loss_weight", "lr_factor", "lr_patience"
        ]

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            rmse_header = [f"rmse_{name}" for name in target_names] if target_names else ["rmse_Valence", "rmse_Arousal", "rmse_Dominance"]
            
            if not file_exists:
                header = ["trial_num", "status", "mean_rmse", "val_loss"] + rmse_header + ["duration_sec"] + fixed_param_keys + ["VAD_weights_normalized"]
                writer.writerow(header)

            rmse_values = [trial.user_attrs.get(f"val_rmse_{name}", "N/A") for name in target_names] if target_names else ["N/A"] * 3
            param_values = [trial.params.get(k, getattr(opt, k, "N/A")) for k in fixed_param_keys]
            row = [
                trial.number,
                status,
                round(mean_rmse, 4) if mean_rmse != float("inf") else "N/A",
                round(best_val_loss, 4) if best_val_loss != float("inf") else "N/A",
            ] + rmse_values + [elapsed_time] + param_values + [str(opt.VAD_weights)]
            
            writer.writerow(row)

    return mean_rmse


def main():
    parser = build_parser()

    # Mode selection for Optuna hyperparameter search
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--study_name", type=str, default="vad_hyperparameter_tune")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_vad.db")
    parser.add_argument("--log_dir", type=str, default="./output/optuna_logs")
    parser.add_argument("--resume_study", action="store_true", help="Resume an existing Optuna study if it exists")

    # Dynamic Range Arguments using custom parsing utils
    parser.add_argument("--range_learning_rate", type=parse_csv_floats, default="1e-4, 1e-1", help="Min, Max bounds for LR")
    parser.add_argument("--range_weight_decay", type=parse_csv_floats, default="1e-6, 1e-2", help="Min, Max bounds for weight decay")
    parser.add_argument("--range_backbone_lr_scale", type=parse_csv_floats, default="1e-3, 1.0", help="Min, Max for backbone LR scaling")
    parser.add_argument("--range_unfreeze_epoch", type=parse_csv_ints, default="0, 15", help="Min, Max unfreeze epoch bounds")
    parser.add_argument("--range_input_size", type=parse_csv_ints, default="112, 224", help="Discrete input resolutions")
    parser.add_argument("--range_batch_size", type=parse_csv_ints, default="16, 32, 64", help="Discrete batch sizes")
    parser.add_argument("--range_dropout_rate", type=parse_csv_floats, default="0.1, 0.6, 0.1", help="Min, Max, Step for dropout")
    parser.add_argument("--range_weights_source", type=parse_csv_strings, default="imagenet, custom", help="Weights source options")
    parser.add_argument("--range_optimizer", type=parse_csv_strings, default="adam, adamw, sgd", help="Optimizers to search over")
    parser.add_argument("--range_criterion", type=parse_csv_strings, default="ccc, mse, combined", help="Criterions to search over")
    parser.add_argument("--range_ccc_weight", type=parse_csv_floats, default="0.1, 0.9, 0.1", help="Min, Max, Step for CCC loss weight")
    parser.add_argument("--range_orth_loss_weight", type=parse_csv_floats, default="0.0, 1.0, 0.1", help="Min, Max, Step for Orth loss weight")
    parser.add_argument("--range_lr_factor", type=parse_csv_floats, default="0.1, 0.5", help="Discrete learning rate decay factors")
    parser.add_argument("--range_lr_patience", type=parse_csv_ints, default="3, 10", help="Min, Max bounds for scheduler patience")

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    log_csv_path = os.path.join(opt.log_dir, f"{opt.study_name}_{opt.dataset}_trials.csv")
    summary_txt_path = os.path.join(opt.log_dir, f"{opt.study_name}_{opt.dataset}_summary.txt")

    sampler = TPESampler(seed=opt.optuna_seed)
    pruner = HyperbandPruner(
        min_resource=3,
        max_resource=getattr(opt, "epochs", 30),
        reduction_factor=3
    )

    if not opt.resume_study:
        try:
            optuna.delete_study(study_name=opt.study_name, storage=opt.storage)
            print(f"Deleted existing study '{opt.study_name}' to start fresh.")
        except Exception:
            pass

        if os.path.exists(log_csv_path):
            os.remove(log_csv_path)

    study = optuna.create_study(
        study_name=opt.study_name,
        storage=opt.storage,
        load_if_exists=opt.resume_study,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    print(f"Dataset targeted: {opt.dataset.upper()}")
    print(f"Starting Optuna Study '{opt.study_name}' ({opt.n_trials} trials). Logging to: {log_csv_path}")

    study.optimize(
        lambda trial: objective(trial, opt, log_csv_path),
        n_trials=opt.n_trials,
        catch=(Exception,),
    )

    summary_content = (
        f"Study Name: {opt.study_name}\n"
        f"Dataset: {opt.dataset}\n"
        f"Total Trials: {len(study.trials)}\n"
        f"Best Mean RMSE: {study.best_value:.5f}\n\n"
        f"Best Hyperparameters:\n"
    )
    for key, value in study.best_params.items():
        summary_content += f"  {key}: {value}\n"

    with open(summary_txt_path, "w") as f:
        f.write(summary_content)

    print("\n" + "=" * 50)
    print("BEST TRIAL PARAMETERS")
    print("=" * 50)
    print(summary_content)
    print(f"Summary log saved to: {summary_txt_path}")


if __name__ == "__main__":
    main()