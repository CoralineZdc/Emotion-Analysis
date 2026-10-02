"""Optuna hyperparameter optimization script."""

import argparse
import copy
import csv
import sys
import os
import gc
import time
from pathlib import Path
from typing import Any, List

import optuna
import torch
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import TPESampler

from src.training.train import build_parser, run_training
from src.utils import (
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
)

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # Suppress C++ CUDACachingAllocator warning logs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"  # Prevent memory thrashing warnings



def get_explicit_cli_args(parser: argparse.ArgumentParser) -> List[str]:
    """
    Extracts explicit parameter names for the training script passed via command line.
    
    Args:
        parser: The argparse.ArgumentParser instance used for parsing CLI arguments.

    Returns:
        A list of explicit parameter dest names.
    """
    explicit_dests = []
    for arg in sys.argv[1:]:
        for action in parser._actions:
            if any(arg == opt_str or arg.startswith(opt_str + "=") for opt_str in action.option_strings):
                if action.dest not in explicit_dests:
                    explicit_dests.append(action.dest)
                break
    return explicit_dests


def build_experiment_tag(opt: argparse.Namespace, explicit_args: List[str]) -> str:
    """
    Dynamically builds an experiment tag from explicitly passed CLI arguments.
    
    Args:
        opt: The argparse.Namespace object containing parsed CLI arguments.
        explicit_args: List of argument dest names that were explicitly passed.

    Returns:
        A string representing the experiment tag for log naming.
    """
    ignore_keys = {"study_name", "storage", "log_dir", "resume_study", "n_trials", "optuna_seed", "pruner"}
    parts = []
    for key in explicit_args:
        # Skip keys that correspond to training parameters set by the user or parameter ranges
        if key in ignore_keys or key.startswith("range_"):
            continue

        val = getattr(opt, key, None)
        if isinstance(val, bool):
            if val:
                parts.append(f"{key}")
            else:
                parts.append(f"not{key}")
        elif val is not None:
            clean_val = str(val).replace(" ", "").replace("/", "").replace("_", "").replace("\\", "")
            parts.append(f"{key}-{clean_val}")

    return f"_{'_'.join(parts)}" if parts else ""



def suggest_param(trial: optuna.trial.Trial, name: str, bounds: List[Any], param_type: str) -> Any:
    """
    Samples a hyperparameter based on its type and length of input range list.
    
    Args:
        trial: The Optuna trial object.
        name: Name of the hyperparameter.
        bounds: List of bounds or options for the hyperparameter.
        param_type: Type of the hyperparameter ('log_float', 'linear_float', 'int', 'categorical').
    Returns:
        The sampled hyperparameter value.
    """
    # Map parameter types to the corresponding Optuna suggestion methods
    suggest_funcs = {
        "log_float": trial.suggest_float,
        "linear_float": trial.suggest_float,
        "int": trial.suggest_int,
        "categorical": trial.suggest_categorical
    }
    suggest = suggest_funcs.get(param_type)
    if suggest is None:
        raise ValueError(f"Unsupported parameter type '{param_type}' for '{name}'.")
    if len(bounds) == 1:
        return bounds[0]

    log = True if param_type == "log_float" else False
    if param_type == "categorical":
        return suggest(name, bounds)
    else:
        if len(bounds) == 2:
            return suggest(name, bounds[0], bounds[1], log=log)
        elif len(bounds) == 3:
            return suggest(name, bounds[0], bounds[1], step=bounds[2])
        else:
            raise ValueError(f"Invalid range specification for {param_type} '{name}': {bounds}. Expecting 1, 2, or 3 values.")


def setup_params(trial: optuna.trial.Trial, base_opt: argparse.Namespace, explicit_args: List[str]) -> argparse.Namespace:
    """
    Sets up hyperparameters for the trial, respecting explicitly passed CLI arguments.
    
    Args:
        trial: The Optuna trial object.
        base_opt: The arguments passed from the command line.
        explicit_args: List of explicitly passed CLI arguments.
    Returns:
        The model training argparse namespace with the sampled hyperparameters.
    """
    opt = copy.deepcopy(base_opt)

    opt.no_model_save = True
    opt.use_amp = True  

    param_types = {
        "categorical": ["model", "weights_source", "input_size", "optimizer", "criterion", "scheduler", "lr_factor"],
        "log_float": ["head_lr", "backbone_lr", "weight_decay"],
        "linear_float": ["dropout_rate"],
        "int": ["unfreeze_epoch", "lr_patience", "lr_warmup_epochs"]
    }

    for type, params in param_types.items():
        for param in params:
            if param not in explicit_args:
                bounds = list(getattr(opt, f"range_{param}", None))
                setattr(opt, param, suggest_param(trial, param, bounds, type))
            else:
                trial.set_user_attr(param, getattr(opt, param))

    # Bounds for batch size are adjusted based on input size to prevent OOM errors
    if "batch_size" not in explicit_args:
        if opt.input_size == 224:
            valid_batch_sizes = [b for b in opt.range_batch_size if b <= 16]
        else:
            valid_batch_sizes = [b for b in opt.range_batch_size if b <= 64]
        if len(valid_batch_sizes) == 0:
            valid_batch_sizes = [8, 16, 32, 64]  # Fallback to safe batch sizes if none are valid
        opt.batch_size = suggest_param(trial, f"batch_size_{opt.input_size}", valid_batch_sizes, "categorical")
    else:
        trial.set_user_attr("batch_size", getattr(opt, "batch_size"))

    # Define VAD weights based on the dataset, ensuring they sum to 1.0 to prevent scaling issues in the loss function
    if "VAD_weights" not in explicit_args:
        w_v = suggest_param(trial, "weight_v", [0.0, 1.0], "linear_float")
        w_a = suggest_param(trial, "weight_a", [0.0, 1.0], "linear_float")

        if opt.dataset.lower() == "afew":
            w_d = 0.0
            total_w = w_v + w_a
        else:
            w_d = suggest_param(trial, "weight_d", [0.0, 1.0], "linear_float")
            total_w = w_v + w_a + w_d

        opt.VAD_weights = [round(w_v / total_w, 4), round(w_a / total_w, 4), round(w_d / total_w, 4)]
    else:
        trial.set_user_attr("VAD_weights", getattr(opt, "VAD_weights"))

    # Define CCC weight only if the criterion is set to 'combined' and it wasn't explicitly passed
    if opt.criterion == "combined":
        if "ccc_weight" not in explicit_args:
            opt.ccc_weight = suggest_param(trial, "ccc_weight", opt.range_ccc_weight, "linear_float")
        else:
            trial.set_user_attr("ccc_weight", getattr(opt, "ccc_weight"))
    else:
        opt.ccc_weight = 0.0

    return opt


def objective(trial: optuna.trial.Trial, base_opt: argparse.Namespace, explicit_args: List[str], log_csv_path: Path) -> float:
    """
    Optuna objective function optimized for Mean RMSE minimization.
    
    Args:
        trial: The Optuna trial object.
        base_opt: The base argparse.Namespace object containing CLI arguments.
        explicit_args: List of training parameters set explicitly.
        log_csv_path: Path to the log CSV file.
    Returns:
        The loss of the best trial.
    """
    opt = copy.deepcopy(base_opt)
    opt = setup_params(trial, opt, explicit_args)

    start_time = time.time()
    status = "COMPLETE"
    best_val_loss = float("inf")
    best_metrics = {}
    target_names = ["Valence", "Arousal"] if getattr(opt, "dataset", "").lower() == "afew" else ["Valence", "Arousal", "Dominance"]

    # Run the training process
    best_metrics = None
    try:
        best_val_loss, best_metrics, returned_targets = run_training(opt, trial=trial)
        if returned_targets:
            target_names = returned_targets

    except optuna.exceptions.TrialPruned:
        status = "PRUNED"
        raise
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        status = "PRUNED (OOM)"
        raise optuna.exceptions.TrialPruned(f"Trial {trial.number} pruned due to CUDA OOM.")
    except Exception as e:
        status = f"FAILED ({str(e)})"
        print(f"[Trial {trial.number}] Encountered exception: {e}")
        raise
    finally:
        elapsed_time = round(time.time() - start_time, 2)
        if type(best_metrics) != dict: 
            best_metrics = trial.user_attrs.get(
                "best_metrics", {f"{metric}_overall": "N/A" for metric in ["mse", "rmse", "ccc"]} | 
                {f"{metric}_per_dim": ["N/A"]*len(target_names) for metric in ["mse", "rmse", "ccc"]} |
                {f"pred_std": ["N/A"]*len(target_names) for dim in target_names}
            )

        # Log trial results to CSV
        file_exists = log_csv_path.exists()
        all_param_keys = [
            "model", "weights_source", "unfreeze_epoch", "input_size", "head_lr", "backbone_lr",
            "dataset", "weight_decay", "optimizer", "batch_size", "dropout_rate", 
            "criterion", "ccc_weight",
            "scheduler", "lr_factor", "lr_patience"
        ]
        active_param_keys = [
            p for p in all_param_keys 
            if p not in explicit_args and 
            not (p == "ccc_weight" and getattr(opt, "criterion", None) != "combined" and "criterion" in explicit_args) and 
            len(getattr(opt, f"range_{p}", []) or []) > 1
        ]
        include_vad_weights = "VAD_weights" not in explicit_args
        metric_headers, metric_values = [], []

        # Log metrics in a structured way, handling both overall and per-dimension metrics
        metric_keys = ["mse", "rmse", "ccc"]
        for metric in metric_keys:
            overall_key = f"{metric}_overall"
            per_dim_key = f"{metric}_per_dim"
            if overall_key in best_metrics:
                metric_headers.append(overall_key)
                metric_values.append(round(best_metrics[overall_key], 5) if isinstance(best_metrics[overall_key], (float, int)) else "N/A")
            if per_dim_key in best_metrics and isinstance(best_metrics[per_dim_key], list):
                for val, target in zip(best_metrics[per_dim_key], target_names):
                    metric_headers.append(f"{metric}_{target}")
                    metric_values.append(round(val, 5) if isinstance(val, (float, int)) else "N/A")
        if "pred_std" in best_metrics and isinstance(best_metrics["pred_std"], list):
            for idx, dim in enumerate(target_names):
                metric_headers.append(f"std_{dim}")
                metric_values.append(round(best_metrics["pred_std"][idx], 5) if isinstance(best_metrics["pred_std"][idx], (float, int)) else "N/A")

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            
            if not file_exists:
                header = ["trial_num", "status", "val_loss"] + metric_headers + ["duration_sec"] + active_param_keys
                if include_vad_weights:
                    header.append("VAD_weights_normalized")
                writer.writerow(header)

            param_values = [trial.params.get(k, getattr(opt, k, "N/A")) for k in active_param_keys]
            row = [
                trial.number,
                status,
                round(best_val_loss, 4) if best_val_loss != float("inf") else "N/A",
            ] + metric_values + [elapsed_time] + param_values

            if include_vad_weights:
                row.append(getattr(opt, "VAD_weights", "N/A"))
            
            writer.writerow(row)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_val_loss


def main():
    parser = build_parser()

    # Mode selection for Optuna hyperparameter search
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--study_name", type=str, default="vad_hyperparameter_tune")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_vad.db")
    parser.add_argument("--log_dir", type=str, default="./output/optuna_logs")
    parser.add_argument("--resume_study", action="store_true", help="Resume an existing Optuna study if it exists")
    parser.add_argument("--pruner", type=str, default="hyperband", choices=["hyperband", "median", "none"], help="Pruner type for Optuna trials")

    # Dynamic Range Arguments using custom parsing utils
    parser.add_argument("--range_dataset", type=parse_csv_strings, default="fer,afew,emotic,emotic-child,heco", help="Datasets to search over")
    parser.add_argument("--range_model", type=parse_csv_strings, default="vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,efficientnet,mobilenet,mobilefacenet", help="Backbone model options")
    parser.add_argument("--range_head_lr", type=parse_csv_floats, default="1e-5, 1e-1", help="'Min, Max' bounds for head LR")
    parser.add_argument("--range_backbone_lr", type=parse_csv_floats, default="1e-5, 1e-1", help="'Min, Max' bounds for backbone LR")
    parser.add_argument("--range_weight_decay", type=parse_csv_floats, default="1e-6, 1.0", help="'Min, Max' bounds for weight decay")
    parser.add_argument("--range_unfreeze_epoch", type=parse_csv_ints, default="0, 15", help="'Min, Max' bounds for unfreeze epoch")
    parser.add_argument("--range_input_size", type=parse_csv_ints, default="112, 224", help="Discrete input resolutions")
    parser.add_argument("--range_batch_size", type=parse_csv_ints, default="16, 32, 64", help="Discrete batch sizes")
    parser.add_argument("--range_dropout_rate", type=parse_csv_floats, default="0.1, 0.6, 0.1", help="'Min, Max, Step' for dropout")
    parser.add_argument("--range_weights_source", type=parse_csv_strings, default="imagenet, custom", help="Weights source options")
    parser.add_argument("--range_optimizer", type=parse_csv_strings, default="adam, adamw, sgd", help="Optimizers to search over")
    parser.add_argument("--range_criterion", type=parse_csv_strings, default="ccc, mse, combined", help="Criterions to search over")
    parser.add_argument("--range_ccc_weight", type=parse_csv_floats, default="0.1, 0.9, 0.1", help="'Min, Max, Step' for CCC loss weight")
    parser.add_argument("--range_scheduler", type=parse_csv_strings, default="reduce_on_plateau, cosine_annealing", help="Schedulers to search over")
    parser.add_argument("--range_lr_factor", type=parse_csv_floats, default="0.1, 0.5", help="Discrete learning rate decay factors")
    parser.add_argument("--range_lr_patience", type=parse_csv_ints, default="3, 10", help="'Min, Max' bounds for scheduler patience")
    parser.add_argument("--range_lr_warmup_epochs", type=parse_csv_ints, default="0, 5", help="'Min, Max' bounds for warmup epochs")

    # Identify which CLI arguments were explicitly passed to ensure they are not overridden by Optuna sampling
    explicit_args = get_explicit_cli_args(parser)
    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)

    # Construct a unique experiment tag based on explicitly passed CLI arguments
    exp_tag = build_experiment_tag(opt, explicit_args)
    full_study_name = f"{opt.study_name}{exp_tag}"
    log_csv_path = Path(opt.log_dir) / f"{full_study_name}_trials.csv"
    summary_txt_path = Path(opt.log_dir) / f"{full_study_name}_{opt.dataset}_summary.txt"

    # Set up Optuna sampler and pruner based on user selection
    sampler = TPESampler(seed=opt.optuna_seed)
    if opt.pruner == "hyperband":
        pruner = HyperbandPruner(
            min_resource=3,
            max_resource=getattr(opt, "epochs", 30),
            reduction_factor=3
        )
    elif opt.pruner == "median":
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        )
    else:
        pruner = NopPruner()

    # If not resuming, delete any existing study and log file to start fresh
    if not opt.resume_study:
        try:
            optuna.delete_study(study_name=full_study_name, storage=opt.storage)
            print(f"Deleted existing study '{full_study_name}' to start fresh.")
        except Exception:
            pass
        if log_csv_path.exists():
            log_csv_path.unlink()

    study = optuna.create_study(
        study_name=full_study_name,
        storage=opt.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    print(f"Dataset targeted: {opt.dataset.upper()}")
    print(f"Starting Optuna Study '{full_study_name}' ({opt.n_trials} trials). Logging to: {log_csv_path}")

    study.optimize(
        lambda trial: objective(trial, opt, explicit_args, log_csv_path),
        n_trials=opt.n_trials,
        catch=(Exception,),
    )

    summary_content = (
        f"Study Name: {full_study_name}\n"
        f"Dataset: {opt.dataset}\n"
        f"Total Trials: {len(study.trials)}\n"
        f"Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} [{len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])/len(study.trials)*100:.1f}%]\n"
        f"Best Loss: {study.best_value:.5f}\n"
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