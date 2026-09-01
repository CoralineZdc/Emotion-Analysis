import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Navigate UP 3 levels: evaluation -> src -> Project Root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.parsing_utils import repo_root


def format_title_from_folder(folder_name: str) -> str:
    """Format experiment folder name into a readable plot title."""
    simu_params_list = folder_name.split("_")
    for id_param, simu_param in enumerate(simu_params_list):
        simu_param = simu_param.strip()
        simu_param = simu_param.replace("-", "=")
        for id_letter, letter in enumerate(simu_param):
            if letter.isdigit():
                simu_params_list[id_param] = simu_param[:id_letter] + "=" + simu_param[id_letter:]
                break

    if len(simu_params_list) % 6 != 0:
        for _ in range(6 - len(simu_params_list) % 6):
            simu_params_list.append("")

    simu_params_str = ""
    for i in range(len(simu_params_list) // 6):
        simu_params_str += ", ".join(simu_params_list[6 * i : 6 * i + 6]).strip(", ") + "\n"

    return simu_params_str.strip()


def plot_loss_curve(csv_path: Path) -> Path:
    """Plot Loss and per-dimension RMSE curves from log CSV."""
    df = pd.read_csv(csv_path)

    if "train_loss" not in df.columns or "val_loss" not in df.columns:
        raise ValueError(f"CSV file {csv_path} must contain 'train_loss' and 'val_loss' columns.")

    plt.figure(figsize=(11, 7))

    # Plot Overall Loss (Black)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="black", linestyle="--", alpha=0.8)
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", color="black", linestyle="-", linewidth=2)

    # Map Dimensions to Colors
    dim_colors = [
        ("Valence", "green"),
        ("Arousal", "red"),
        ("Dominance", "blue"),
    ]

    # Plot Registered Dimension RMSEs
    for dim_name, color in dim_colors:
        train_col = f"{dim_name}_train_rmse"
        val_col = f"{dim_name}_val_rmse"

        if train_col in df.columns:
            plt.plot(df["epoch"], df[train_col], label=f"{dim_name} Train RMSE", color=color, linestyle="--", alpha=0.7)
        if val_col in df.columns:
            plt.plot(df["epoch"], df[val_col], label=f"{dim_name} Val RMSE", color=color, linestyle="-", linewidth=1.8)

    # Styling
    title_str = format_title_from_folder(csv_path.parent.name)
    plt.title(title_str, fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    # Save to the specific experiment directory containing log.csv
    output_file = csv_path.parent / "loss_curve.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    return output_file


def discover_csvs(search_dir: Path) -> list[Path]:
    """Recursively find all log.csv files in subdirectories."""
    return sorted(
        path for path in Path(search_dir).rglob("*.csv")
        if path.is_file() and not path.name.startswith("optuna_")  # Skip optuna search logs
    )


def run(args: argparse.Namespace) -> tuple[int, list[str]]:
    search_dir = Path(args.dir)
    repo_root_path = Path(repo_root())

    if not search_dir.is_absolute():
        search_dir = repo_root_path / search_dir

    csv_paths = discover_csvs(search_dir)

    if not csv_paths:
        raise SystemExit(f"No CSV files found in {search_dir}")

    saved = 0
    skipped: list[str] = []

    for csv_path in csv_paths:
        try:
            out_path = plot_loss_curve(csv_path)
            print(f"Saved: {out_path}")
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