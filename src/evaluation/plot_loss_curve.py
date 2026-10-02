"""Script to plot training and validation loss/metrics curves from experiment CSV logs."""

import argparse
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import get_simu_params, get_project_root


def format_title_from_folder(folder_path: Union[str, Path]) -> str:
    """
    Dynamically format experiment folder name into a multi-line plot title.
    
    Args:
        folder_path: Path to the experiment folder.

    Returns:
        Formatted title string.
    """
    folder_path = Path(folder_path)
    params = get_simu_params(folder_path)
    if not params:
        return folder_path.name  # Return the folder name as-is if no parameters are found
    
    formatted_params = [f"{key}={value}" for key, value in params.items()]

    # Wrap parameters evenly across lines (5 items per line)
    chunk_size = 5
    lines = [
        ", ".join(formatted_params[i : i + chunk_size])
        for i in range(0, len(formatted_params), chunk_size)
    ]
    return "\n".join(lines)


def plot_loss_curve(csv_path: Path) -> Path:
    """Plots Overall Loss, per-dimension RMSE, and per-dimension CCC from a log.csv file.

    Args:
        csv_path: Path to the log.csv file.

    Returns:
        Path to saved loss_curve.png image.
    """
    df = pd.read_csv(csv_path)

    if "train_loss" not in df.columns or "val_loss" not in df.columns:
        raise ValueError(f"CSV file {csv_path} must contain 'train_loss' and 'val_loss' columns.")

    fig, axs = plt.subplots(3, 1, figsize=(11, 14), sharex=True, layout="constrained")

    linestyles = {"train": "--", "val": "-"}
    dim_colors = {"Valence": "green", "Arousal": "red", "Dominance": "blue"}

    # Plot overall loss and per-dimension metrics for both train and validation splits
    for split, linestyle in linestyles.items():
        col_name = f"{split}_loss"
        if col_name in df.columns:
            axs[0].plot(df["epoch"], df[col_name], label=f"{split.capitalize()} Loss", color="black", linestyle=linestyle)
            for idx, loss_type in enumerate(["rmse", "ccc"]):
                for dim_name, color in dim_colors.items():
                    col_name = f"{split}_{loss_type}_{dim_name}"
                    if col_name in df.columns:
                        axs[idx+1].plot(df["epoch"], df[col_name], label=f"{dim_name[0]} ({split.capitalize()})", color=color, linestyle=linestyle)

    titles = ["Overall Loss", "RMSE per Dimension", "CCC per Dimension"]
    y_labels = ["Loss", "RMSE", "CCC"]

    for idx in range(3):
        axs[idx].set_title(titles[idx], fontsize=11, fontweight="bold")
        axs[idx].set_ylabel(y_labels[idx])
        axs[idx].set_xlabel("Epoch")
        axs[idx].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0, fontsize=9)
        axs[idx].grid(True, linestyle="--", alpha=0.5)

    title_str = format_title_from_folder(csv_path)
    fig.suptitle(title_str, fontsize=12, fontweight="bold")

    # Save to the specific experiment directory containing log.csv
    output_file = csv_path.parent / "loss_curve.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def discover_csvs(search_path: Path) -> List[Path]:
    """
    Recursively discovers log.csv files, skipping optuna trial logs.
    
    Args:
        search_path: Directory or file path to search for CSV logs.

    Returns:
        List of discovered CSV file paths.
    """
    if search_path.is_file():
        return [search_path] if search_path.name == "log.csv" or search_path.suffix == ".csv" else []

    return sorted(
        path for path in search_path.rglob("*.csv")
        if path.is_file() and not path.name.startswith("optuna_") and not path.name.endswith("_trials.csv")
    )


def run(args: argparse.Namespace) -> Tuple[int, List[str]]:
    """
    Executes loss curve plotting for all discovered CSV files in the specified directory.
    
    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (number of plots saved, list of skipped files).
    """
    search_dir = Path(args.dir)
    project_root_path = Path(get_project_root())

    if not search_dir.is_absolute():
        search_dir = project_root_path / search_dir

    csv_paths = discover_csvs(search_dir)
    if not csv_paths:
        raise SystemExit(f"No valid log CSV files found in {search_dir}")

    saved = 0
    skipped: list[str] = []

    for csv_path in csv_paths:
        try:
            out_path = plot_loss_curve(csv_path)
            print(f"Generated plot: {out_path}")
            saved += 1
        except Exception as exc:
            skipped.append(f"{csv_path}: {exc}")

    return saved, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Train vs Validation Loss and VAD RMSE curves from CSV logs.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./output",
        help="Directory containing log.csv files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    saved, skipped = run(args)

    print(f"\nSuccessfully generated {saved} plot(s).")
    if skipped:
        print("Skipped files:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()