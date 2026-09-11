import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_splits(data_dir: Path, dataset_name: str, age_suffix: str | None = None) -> pd.DataFrame:
    """Loads train, val, and test CSV files into a unified DataFrame."""
    splits = ["train", "val", "test"]
    dfs = []

    for split in splits:
        filename = f"{split}-{dataset_name.lower()}-{age_suffix}.csv" if age_suffix else f"{split}-{dataset_name.lower()}.csv"
        filepath = data_dir / filename

        if not filepath.exists():
            print(f"Warning: File not found -> {filepath}")
            continue

        df = pd.read_csv(filepath)
        df["Split"] = split.capitalize()
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No split files found for dataset '{dataset_name}' in '{data_dir}'.")

    return pd.concat(dfs, ignore_index=True)


def plot_1d_density_comparison(df: pd.DataFrame, dataset_title: str, save_path: Path | None = None):
    """Overlays 1D KDE density curves for each VAD dimension across Train, Val, and Test."""
    dimensions = [col for col in ["Valence", "Arousal", "Dominance"] if col in df.columns]
    palette = {"Train": "#1f77b4", "Val": "#ff7f0e", "Test": "#2ca02c"}

    fig, axes = plt.subplots(1, len(dimensions), figsize=(5 * len(dimensions), 4.5), constrained_layout=True)
    if len(dimensions) == 1:
        axes = [axes]

    fig.suptitle(f"1D Label Density Distribution - {dataset_title}", fontsize=14, fontweight="bold")

    for ax, dim in zip(axes, dimensions):
        sns.kdeplot(
            data=df,
            x=dim,
            hue="Split",
            palette=palette,
            common_norm=False,
            fill=True,
            alpha=0.25,
            linewidth=2,
            ax=ax,
        )
        ax.set_title(f"{dim} Distribution")
        ax.set_xlabel(dim)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        out_path = save_path / f"{dataset_title.lower()}_1d_density.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved 1D Density plot -> {out_path}")
    plt.show()


def plot_pairwise_2d_matrix(df: pd.DataFrame, dataset_title: str, save_path: Path | None = None):
    """Plots 2D scatter and contour projections for (V vs A), (V vs D), and (A vs D) per split."""
    dimensions = [col for col in ["Valence", "Arousal", "Dominance"] if col in df.columns]
    if len(dimensions) < 2:
        return

    pairs = [("Valence", "Arousal")]
    if "Dominance" in dimensions:
        pairs.extend([("Valence", "Dominance"), ("Arousal", "Dominance")])

    splits = ["Train", "Val", "Test"]
    palette = {"Train": "#1f77b4", "Val": "#ff7f0e", "Test": "#2ca02c"}

    fig, axes = plt.subplots(
        len(pairs), len(splits), figsize=(4.5 * len(splits), 4 * len(pairs)), sharex="row", sharey="row", constrained_layout=True
    )

    if len(pairs) == 1:
        axes = [axes]

    fig.suptitle(f"Pairwise 2D VAD Projections - {dataset_title}", fontsize=16, fontweight="bold")

    for row_idx, (x_dim, y_dim) in enumerate(pairs):
        for col_idx, split in enumerate(splits):
            ax = axes[row_idx][col_idx]
            split_df = df[df["Split"] == split]

            if split_df.empty:
                ax.set_title(f"{split} (No Data)")
                continue

            # Density contours + light background scatter
            sns.kdeplot(
                data=split_df,
                x=x_dim,
                y=y_dim,
                ax=ax,
                color=palette[split],
                alpha=0.5,
                levels=5,
                fill=True,
            )
            sns.scatterplot(
                data=split_df,
                x=x_dim,
                y=y_dim,
                ax=ax,
                color=palette[split],
                alpha=0.25,
                s=10,
                edgecolor=None,
            )

            if row_idx == 0:
                ax.set_title(f"{split} Split (n={len(split_df)})", fontweight="bold")
            ax.set_xlabel(x_dim)
            ax.set_ylabel(y_dim)
            ax.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        out_path = save_path / f"{dataset_title.lower()}_pairwise_2d.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved Pairwise 2D plot -> {out_path}")
    plt.show()


def print_summary_stats(df: pd.DataFrame):
    """Prints count, mean, std, and quantiles per split."""
    print("\n=== Split Statistical Summary ===")
    cols = [c for c in ["Valence", "Arousal", "Dominance"] if c in df.columns]
    summary = df.groupby("Split")[cols].agg(["count", "mean", "std"])
    print(summary.to_string())
    print("=" * 35 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize VAD distributions across train/val/test splits.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing split CSVs")
    parser.add_argument("--dataset", type=str, required=True, choices=["afew", "emotic", "heco"], help="Dataset to visualize")
    parser.add_argument("--target_age", type=str, default=None, choices=["child", "adult"], help="Age suffix filter if applicable")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save generated plot images")

    args = parser.parse_args()
    data_path = Path(args.data_dir)
    save_path = Path(args.save_dir) if args.save_dir else Path("./output/data")
    save_path.mkdir(parents=True, exist_ok=True)

    df = load_splits(data_path, args.dataset, age_suffix=args.target_age)
    
    dataset_title = f"{args.dataset.upper()}-{args.target_age.upper()}" if args.target_age else args.dataset.upper()
    print_summary_stats(df)

    # 1. Plot 1D Overlaid Density (Immediate split alignment check)
    plot_1d_density_comparison(df, dataset_title, save_path=save_path)

    # 2. Plot Pairwise 2D Matrix (Replaces hard-to-read 3D scatter)
    plot_pairwise_2d_matrix(df, dataset_title, save_path=save_path)


if __name__ == "__main__":
    main()