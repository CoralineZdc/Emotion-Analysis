import argparse
import copy
import csv
import os
import time
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.training.train import build_parser, run_training


def objective(trial: optuna.trial.Trial, base_opt: argparse.Namespace, log_csv_path: str) -> float:
    """Optuna objective function with detailed CSV logging."""
    opt = copy.deepcopy(base_opt)

    # 1. Hyperparameter Search Space
    #opt.model = trial.suggest_categorical("model", ["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "efficientnet", "mobilenet", "mobilefacenet"])
    opt.pretrained = True
    opt.freezed = trial.suggest_categorical("freezed", [True, False])
    opt.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    opt.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    opt.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    opt.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    opt.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6, step=0.1)

    w_v = trial.suggest_float("weight_v", 0.0, 1.0, step=0.1)
    w_a = trial.suggest_float("weight_a", 0.0, 1.0, step=0.1)

    if opt.dataset.lower() == "afew": w_d = 0.0
    else: w_d = trial.suggest_float("weight_d", 0.0, 1.0, step=0.1)

    opt.VAD_weights = [w_v, w_a, w_d]

    opt.criterion = trial.suggest_categorical("criterion", ["ccc", "mse", "combined"])
    if opt.criterion == "combined":
        opt.ccc_weight = trial.suggest_float("ccc_weight", 0.1, 0.9, step=0.1)

    opt.orth_loss_weight = trial.suggest_float("orth_loss_weight", 0.0, 1.0, step=0.1)
    opt.lr_factor = trial.suggest_categorical("lr_factor", [0.1, 0.5])
    opt.lr_patience = trial.suggest_int("lr_patience", 5, 15)

    # Prevent disk fill-up during hyperparameter search
    opt.no_checkpoint = True
    opt.no_model_save = True

    start_time = time.time()
    status = "COMPLETE"
    best_val_loss = float("inf")
    best_rmse_per_dim = None
    target_names = []

    try:
        best_val_loss, best_rmse_per_dim, target_names = run_training(opt, trial=trial)
        if best_rmse_per_dim is not None:
            for name, rmse in zip(target_names, best_rmse_per_dim):
                rmse_float = float(rmse.item()) if hasattr(rmse, "item") else float(rmse)
                trial.set_user_attr(f"val_rmse_{name}", rmse_float)
    except optuna.exceptions.TrialPruned:
        status = "PRUNED"
        raise
    except Exception as e:
        status = f"FAILED ({str(e)})"
        print(f"[Trial {trial.number}] Encountered exception: {e}")
        return float("inf")
    finally:
        elapsed_time = round(time.time() - start_time, 2)

        rmse_dict = {}
        if best_rmse_per_dim is not None and len(target_names) == len(best_rmse_per_dim):
            rmse_dict = {f"rmse_{name}": round(float(rmse), 4) for rmse, name in zip(best_rmse_per_dim, target_names)}
        
        # Save trial summary log entry to CSV
        file_exists = os.path.exists(log_csv_path)
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            rmse_header = [f"rmse_{name}" for name in target_names] if target_names else ["rmse_Valence", "rmse_Arousal"]
            param_keys = list(trial.params.keys())

            if not file_exists:
                # Write header on first trial
                header = ["trial_num", "status", "val_loss"] + rmse_header + ["duration_sec"] + list(trial.params.keys())
                writer.writerow(header)
            
            rmse_values = [rmse_dict.get(h, "N/A") for h in rmse_header]
            row = [trial.number, status, best_val_loss if best_val_loss != float("inf") else "N/A"] + rmse_values + [elapsed_time] + list(trial.params.values())
            writer.writerow(row)

    return best_val_loss


def main():
    parser = build_parser()
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--study_name", type=str, default="afew_hyperparameter_tune")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_vad.db")
    parser.add_argument("--log_dir", type=str, default="./output/optuna_logs")
    parser.add_argument("--resume_study", action="store_true", help="Resume an existing Optuna study if it exists")

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    log_csv_path = os.path.join(opt.log_dir, f"{opt.study_name}_trials.csv")
    summary_txt_path = os.path.join(opt.log_dir, f"{opt.study_name}_summary.txt")

    sampler = TPESampler(seed=opt.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    if not opt.resume_study:
        try:
            optuna.delete_study(study_name=opt.study_name, storage=opt.storage)
            print(f"Deleted existing study '{opt.study_name}' to start fresh.")
        except Exception:
            pass

        if os.path.exists(log_csv_path):
            os.remove(log_csv_path)
            print(f"Deleted existing log file '{log_csv_path}' to start fresh.")

    study = optuna.create_study(
        study_name=opt.study_name,
        storage=opt.storage,
        load_if_exists=opt.resume_study,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    mode_str = "Resuming" if opt.resume_study else "Starting"
    print(f"{mode_str} Optuna Study '{opt.study_name}' with {opt.n_trials} trials. \nLogging to: {log_csv_path}")

    study.optimize(
        lambda trial: objective(trial, opt, log_csv_path),
        n_trials=opt.n_trials,
        catch=(Exception,),
    )

    # Save summary log upon study completion
    summary_content = (
        f"Study Name: {opt.study_name}\n"
        f"Dataset: {opt.dataset}\n"
        f"Total Trials: {len(study.trials)}\n"
        f"Best Validation Loss: {study.best_value:.5f}\n\n"
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