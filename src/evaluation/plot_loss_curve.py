import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


from src.utils.parsing_utils import repo_root, get_simu_params


def format_title_from_folder(folder_name: str) -> str:
    """Dynamically format experiment folder name into a multi-line plot title."""
    params = get_simu_params(Path(folder_name))
    formatted_params = [f"{key}={value}" for key, value in params.items()]

    # Wrap parameters evenly across lines (5 items per line)
    chunk_size = 5
    lines = [
        ", ".join(formatted_params[i : i + chunk_size])
        for i in range(0, len(formatted_params), chunk_size)
    ]
    return "\n".join(lines)


def plot_loss_curve(csv_path: Path) -> Path:
    """Plot Loss and per-dimension RMSE curves from log CSV."""
    df = pd.read_csv(csv_path)

    if "train_loss" not in df.columns or "val_loss" not in df.columns:
        raise ValueError(f"CSV file {csv_path} must contain 'train_loss' and 'val_loss' columns.")

    linestyles = {"train": "--", "val": "-"}
    dim_colors = {"Valence": "green", "Arousal": "red", "Dominance": "blue"}

    fig, axs = plt.subplots(3, 1, figsize=(11, 17), sharex=True)

    # Plot Overall Loss (Black)
    for split, linestyle in linestyles.items():
        col_name = f"{split}_loss"
        if col_name in df.columns:
            axs[0].plot(df["epoch"], df[col_name], label=f"{split.capitalize()} Loss", color="black", linestyle=linestyle)
            for idx, loss_type in enumerate(["rmse", "ccc"]):
                for dim_name, color in dim_colors.items():
                    col_name = f"{split}_{loss_type}_{dim_name}"
                    if col_name in df.columns:
                        axs[idx+1].plot(df["epoch"], df[col_name], label=f"{dim_name} {split.capitalize()} {loss_type.upper()}", color=color, linestyle=linestyle)

    for idx, loss_type in enumerate(["Overall Loss", "RMSE", "CCC"]):
        axs[idx].set_ylabel(loss_type)
        axs[idx].set_xlabel("Epoch")
        axs[idx].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        axs[idx].set_title(f"{loss_type}{' per Dimension' if idx > 0 else ''}", fontsize=12)
        axs[idx].grid(True, linestyle="--", alpha=0.5)

    title_str = format_title_from_folder(csv_path.parent.name)
    fig.suptitle(title_str, fontsize=10)
    fig.tight_layout()

    # Save to the specific experiment directory containing log.csv
    output_file = csv_path.parent / "loss_curve.png"
    fig.savefig(output_file, dpi=300)
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