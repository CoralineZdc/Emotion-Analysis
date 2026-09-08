import argparse
import copy
import csv
import os
import time
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.training.train import build_parser, run_training


def objective(trial: optuna.trial.Trial, base_opt: argparse.Namespace, log_csv_path: str) -> float:
    """Optuna objective function optimized for Mean RMSE minimization."""
    opt = copy.deepcopy(base_opt)

    # ---------------------------------------------------------
    # 1. Hyperparameter Search Space
    # ---------------------------------------------------------
    #opt.model = trial.suggest_categorical("model", ["vgg19", "resnet50"])
    opt.pretrained = True
    opt.weights_source = trial.suggest_categorical("weights_source", ["imagenet", "custom"])
    opt.freezed = True
    opt.unfreeze_epoch = trial.suggest_int("unfreeze_epoch", 0, 15)
    opt.input_size = trial.suggest_categorical("input_size", [48, 112])
    
    opt.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    opt.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    opt.backbone_lr_scale = trial.suggest_float("backbone_lr_scale", 1e-3, 1.0, log=True)
    opt.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    opt.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    opt.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6, step=0.1)

    # VAD Weight Normalization (Ensures sum equals 1.0 to prevent loss tricking)
    w_v = trial.suggest_float("weight_v", 0.1, 1.0)
    w_a = trial.suggest_float("weight_a", 0.1, 1.0)
    
    if opt.dataset.lower() == "afew":
        w_d = 0.0
        total_w = w_v + w_a
    else:
        w_d = trial.suggest_float("weight_d", 0.1, 1.0)
        total_w = w_v + w_a + w_d

    opt.VAD_weights = [round(w_v / total_w, 4), round(w_a / total_w, 4), round(w_d / total_w, 4)]

    # Loss Criterion Configuration
    opt.criterion = trial.suggest_categorical("criterion", ["ccc", "mse", "combined"])
    if opt.criterion == "combined":
        opt.ccc_weight = trial.suggest_float("ccc_weight", 0.1, 0.9, step=0.1)
    else:
        opt.ccc_weight = 0.0

    opt.orth_loss_weight = trial.suggest_float("orth_loss_weight", 0.0, 1.0, step=0.1)
    opt.lr_factor = trial.suggest_categorical("lr_factor", [0.1, 0.5])
    opt.lr_patience = trial.suggest_int("lr_patience", 3, 10)

    # Disable disk I/O saving during optimization
    opt.no_checkpoint = True
    opt.no_model_save = True

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
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--optuna_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--study_name", type=str, default="vad_hyperparameter_tune")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_vad.db")
    parser.add_argument("--log_dir", type=str, default="./output/optuna_logs")
    parser.add_argument("--resume_study", action="store_true", help="Resume an existing Optuna study if it exists")

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