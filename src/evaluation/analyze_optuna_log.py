import argparse
import ast
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


class Tee:
    """Helper class to write output simultaneously to stdout and a report file."""
    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def discover_optuna_csvs(search_dir: Path) -> list[Path]:
    """Recursively find all Optuna trial CSV files in subdirectories, sorted by modification time (latest first)."""
    if not search_dir.exists():
        return []
    
    csv_files = [
        path for path in search_dir.rglob("*.csv")
        if path.is_file() and ("optuna" in path.name.lower() or path.name.endswith("_trials.csv"))
    ]
    
    return sorted(csv_files, key=lambda p: p.stat().st_mtime, reverse=True)


def parse_and_normalize_vad(df: pd.DataFrame) -> pd.DataFrame:
    """Detects VAD weight parameters, creates a single 'VAD_weights_normalized' column formatted as [V, A, D],
    and stores internal normalized vector values (_vad_pV, _vad_pA, _vad_pD) for joint PERMANOVA analysis.
    """
    vad_col_name = None
    for col in df.columns:
        if "vad" in col.lower() and "weight" in col.lower():
            vad_col_name = col
            break

    # Parse stringified vector/list columns (e.g., "[1.0, 0.5, 0.2]")
    if vad_col_name and df[vad_col_name].dtype == object:
        def extract_norm(val):
            if pd.isna(val):
                return None
            val_str = str(val).strip()
            if val_str.startswith(("[", "(")) and val_str.endswith(("]", ")")):
                try:
                    parsed = ast.literal_eval(val_str)
                    if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                        arr = np.array(parsed, dtype=float)
                        total = arr.sum()
                        if total > 0:
                            norm_arr = arr / total
                            return [round(float(x), 4) for x in norm_arr]
                except Exception:
                    return None
            return None

        parsed_series = df[vad_col_name].apply(extract_norm)
        df["VAD_weights_normalized"] = parsed_series.apply(lambda x: str(x) if x is not None else np.nan)
        df["_vad_pV"] = parsed_series.apply(lambda x: x[0] if x is not None else np.nan)
        df["_vad_pA"] = parsed_series.apply(lambda x: x[1] if x is not None else np.nan)
        df["_vad_pD"] = parsed_series.apply(lambda x: x[2] if x is not None else np.nan)

    # Fallback: Parse individual weight_v, weight_a, weight_d columns if present
    elif all(c in df.columns for c in ["weight_v", "weight_a", "weight_d"]):
        total = df["weight_v"] + df["weight_a"] + df["weight_d"]
        pV = df["weight_v"] / total
        pA = df["weight_a"] / total
        pD = df["weight_d"] / total

        df["_vad_pV"] = pV
        df["_vad_pA"] = pA
        df["_vad_pD"] = pD

        df["VAD_weights_normalized"] = [
            f"[{pV.iloc[i]:.4f}, {pA.iloc[i]:.4f}, {pD.iloc[i]:.4f}]" if pd.notna(total.iloc[i]) else np.nan
            for i in range(len(df))
        ]

    return df


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Loads CSV log, extracts normalized VAD weights, merges individual VAD RMSEs into 'rmse_VAD', and coerces numeric types."""
    df = pd.read_csv(csv_path)
    df = parse_and_normalize_vad(df)

    numeric_candidates = [
        "mean_rmse", "val_loss", "duration_sec", "batch_size", "orth_loss_weight", "ccc_weight",
        "learning_rate", "weight_decay", "backbone_lr_scale", "dropout_rate",
        "unfreeze_epoch", "lr_factor", "lr_patience", "input_size",
        "rmse_Valence", "rmse_Arousal", "rmse_Dominance"
    ]
    
    for col in df.columns:
        if col in numeric_candidates or any(k in col.lower() for k in ["loss", "lr", "rate", "epoch", "size", "batch"]):
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Combine individual Valence/Arousal/Dominance RMSEs into a concise single column: "V; A; D"
    vad_rmse_cols = ["rmse_Valence", "rmse_Arousal", "rmse_Dominance"]
    if all(c in df.columns for c in vad_rmse_cols):
        df["rmse_VAD"] = df.apply(
            lambda r: f"{format_num(r['rmse_Valence'])}; {format_num(r['rmse_Arousal'])}; {format_num(r['rmse_Dominance'])}"
            if pd.notna(r["rmse_Valence"]) else "N/A",
            axis=1
        )

    return df


def get_parameter_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Includes all parameter columns that vary across trials, excluding fixed or non-informative columns. Returns two lists: categorical and numeric parameters."""
    exclude = {
        "trial_num", "datetime_start", "datetime_complete", "state", "status",
        "duration_sec", "mean_rmse", "val_loss", "rmse_Valence", "rmse_Arousal", "rmse_Dominance", "rmse_VAD",
        "rank_score", "log_rank_score", "is_pruned", "weight_v", "weight_a", "weight_d",
        "weights_v", "weights_a", "weights_d", "_vad_pV", "_vad_pA", "_vad_pD",
        "vad_weights", "VAD_weights_normalized"
    }

    target_numerics = [
        "batch_size", "orth_loss_weight", "ccc_weight", "learning_rate", "weight_decay",
        "backbone_lr_scale", "dropout_rate", "unfreeze_epoch", "lr_factor", "lr_patience"
    ]
    target_categoricals = ["criterion", "optimizer", "input_size", "weights_source"]

    cat_cols = []
    num_cols = []

    for col in df.columns:
        if col in exclude or col.startswith("_vad_") or col.startswith("VAD_w_normalized"):
            continue

        if col in target_categoricals and df[col].nunique() > 1:
            cat_cols.append(col)
        elif col in target_numerics and df[col].nunique() > 1:
            num_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
            num_cols.append(col)
        elif not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
            cat_cols.append(col)

    return sorted(list(set(cat_cols))), sorted(list(set(num_cols)))


def format_num(val):
    """Utility helper for consistent number formatting in output tables."""
    if pd.isna(val):
        return "N/A"
    if isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()):
        return f"{int(val)}"
    if abs(val) < 0.001 or abs(val) > 1000:
        return f"{val:.2e}"
    return f"{val:.4f}"


def format_time(seconds: float) -> str:
    """Formats duration in seconds into readable format."""
    if pd.isna(seconds):
        return "N/A"
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s ({seconds:.1f}s)"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m ({seconds:.1f}s)"


def run_joint_vad_analysis(df: pd.DataFrame):
    """Performs multivariate PERMANOVA and optimal ratio analysis exclusively on normalized VAD weight vectors."""
    vad_cols = ["_vad_pV", "_vad_pA", "_vad_pD"]
    if not all(c in df.columns for c in vad_cols):
        return

    eval_df = df[df["status"].isin(["COMPLETE", "PRUNED"])].dropna(subset=vad_cols).copy()
    if len(eval_df) < 5:
        return

    print("\n[ JOINT VAD WEIGHT VECTOR ANALYSIS (Normalized Proportions Only) ]")
    completed_df = eval_df[eval_df["status"] == "COMPLETE"]

    if len(completed_df) > 0:
        top_n = max(3, int(np.ceil(len(completed_df) * 0.25)))
        top_df = completed_df.sort_values(by="mean_rmse", ascending=True).head(top_n)

        mean_v = top_df["_vad_pV"].mean()
        mean_a = top_df["_vad_pA"].mean()
        mean_d = top_df["_vad_pD"].mean()

        print("Optimal Joint Normalized Proportion Ratio (Top 25% Best Runs):")
        print(f"  Valence : {mean_v*100:.1f}%  |  Arousal : {mean_a*100:.1f}%  |  Dominance : {mean_d*100:.1f}%")
        print(f"  Normalized Vector (V:A:D) = [{mean_v:.4f}, {mean_a:.4f}, {mean_d:.4f}]")

    if eval_df["status"].nunique() > 1:
        X = eval_df[vad_cols].values
        groups = (eval_df["status"] == "PRUNED").astype(int).values

        dist_matrix = squareform(pdist(X, metric="euclidean"))
        idx_pruned = np.where(groups == 1)[0]
        idx_comp = np.where(groups == 0)[0]

        if len(idx_pruned) > 1 and len(idx_comp) > 1:
            def compute_f_stat(d_mat, g1, g2):
                n_tot = len(g1) + len(g2)
                ss_total = np.sum(d_mat**2) / (2 * n_tot)
                ss_within = (np.sum(d_mat[np.ix_(g1, g1)]**2) / (2 * len(g1))) + (np.sum(d_mat[np.ix_(g2, g2)]**2) / (2 * len(g2)))
                ss_between = ss_total - ss_within
                return (ss_between / 1) / (ss_within / (n_tot - 2)) if ss_within > 0 else 0.0

            obs_f = compute_f_stat(dist_matrix, idx_pruned, idx_comp)

            n_perm = 999
            perm_f = []
            np.random.seed(42)
            for _ in range(n_perm):
                shuffled_g = np.random.permutation(groups)
                g1_p = np.where(shuffled_g == 1)[0]
                g2_p = np.where(shuffled_g == 0)[0]
                perm_f.append(compute_f_stat(dist_matrix, g1_p, g2_p))

            p_val = (np.sum(np.array(perm_f) >= obs_f) + 1) / (n_perm + 1)
            sig = "***" if p_val <= 0.01 else ("**" if p_val <= 0.05 else ("*" if p_val <= 0.1 else "ns"))
            print(f"PERMANOVA Joint Normalized VAD Prune Risk Test p-value : {p_val:.4f} [{sig}]")


def print_categorical_table(df: pd.DataFrame, cat_cols: list):
    """Prints clear categorical summary table including prune rates."""
    if not cat_cols:
        return

    header  = "+-------------------+----------------+-------+-----------+-----------+-----------+------------+"
    divider = "+-------------------+----------------+-------+-----------+-----------+-----------+------------+"
    
    print("\n[ CATEGORICAL & DISCRETE PARAMETER SUMMARY ]")
    print(header)
    print(f"| {'Parameter':<17} | {'Category Value':<14} | {'Total':<5} | {'Mean RMSE':<9} | {'Min RMSE':<9} | {'Std Dev':<9} | {'Prune %':<10} |")
    print(header)

    for col in cat_cols:
        if col not in df.columns:
            continue
        
        completed = df[df["status"] == "COMPLETE"]
        group_comp = completed.groupby(col)["mean_rmse"].agg(["count", "mean", "min", "std"]).reset_index()
        group_all = df.groupby(col)["status"].agg(total="count", pruned=lambda s: (s == "PRUNED").sum()).reset_index()
        
        merged = pd.merge(group_all, group_comp, on=col, how="left").sort_values(by="mean", ascending=True).reset_index(drop=True)
        
        for i, row in merged.iterrows():
            param_name = col if i == 0 else ""
            mean_str = f"{row['mean']:.4f}" if pd.notna(row['mean']) else "N/A"
            min_str  = f"{row['min']:.4f}" if pd.notna(row['min']) else "N/A"
            std_str  = f"{row['std']:.4f}" if pd.notna(row['std']) else "N/A"
            
            prune_rate = (row["pruned"] / row["total"]) * 100 if row["total"] > 0 else 0.0
            prune_str = f"{prune_rate:.1f}% ({int(row['pruned'])})"
            
            print(f"| {param_name:<17} | {str(row[col]):<14} | {int(row['total']):<5} | {mean_str:<9} | {min_str:<9} | {std_str:<9} | {prune_str:<10} |")
        
        print(divider)


def run_prune_propensity_test(df: pd.DataFrame, cat_cols: list, num_cols: list):
    """Identifies which parameter values directly trigger trial pruning."""
    print("\n[ PRUNE PROPENSITY ANALYSIS (Parameters Triggering Pruning) ]")
    
    eval_df = df[df["status"].isin(["COMPLETE", "PRUNED"])].copy()
    if len(eval_df) < 5 or eval_df["status"].nunique() < 2:
        print("Requires both completed and pruned trials to compute prune propensity.")
        return

    eval_df["is_pruned"] = (eval_df["status"] == "PRUNED").astype(int)
    results = []

    for col in cat_cols:
        if col in eval_df.columns and eval_df[col].nunique() > 1:
            contingency = pd.crosstab(eval_df[col], eval_df["is_pruned"])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                _, p_val, _, _ = stats.chi2_contingency(contingency)
                
                prune_rates = eval_df.groupby(col)["is_pruned"].mean()
                worst_cat = prune_rates.idxmax()
                highest_rate = prune_rates.max() * 100

                results.append({
                    "Parameter": col,
                    "Type": "Categorical",
                    "Prune Test": "Chi-Square",
                    "p-value": p_val,
                    "Prune Trigger Risk": f"Value '{worst_cat}' ({highest_rate:.0f}% pruned)"
                })

    for col in num_cols:
        if col in eval_df.columns and eval_df[col].nunique() > 1:
            pruned_vals = eval_df[eval_df["is_pruned"] == 1][col].dropna()
            completed_vals = eval_df[eval_df["is_pruned"] == 0][col].dropna()

            if len(pruned_vals) > 1 and len(completed_vals) > 1:
                _, p_val = stats.mannwhitneyu(pruned_vals, completed_vals)
                
                med_pruned = pruned_vals.median()
                med_comp = completed_vals.median()
                direction = "High values" if med_pruned > med_comp else "Low values"

                results.append({
                    "Parameter": col,
                    "Type": "Numeric",
                    "Prune Test": "Mann-Whitney U",
                    "p-value": p_val,
                    "Prune Trigger Risk": f"{direction} (Med: {format_num(med_pruned)})"
                })

    if not results:
        print("No parameter variance detected across trial status outcomes.")
        return

    p_values = [r["p-value"] for r in results if pd.notna(r["p-value"])]
    if p_values:
        q1 = np.percentile(p_values, 25)
        q2 = np.percentile(p_values, 50)
        q3 = np.percentile(p_values, 75)

        for r in results:
            p = r["p-value"]
            if pd.isna(p):
                r["Prune Signif."] = "N/A"
            elif p <= q1:
                r["Prune Signif."] = "***"
            elif p <= q2:
                r["Prune Signif."] = "**"
            elif p <= q3:
                r["Prune Signif."] = "*"
            else:
                r["Prune Signif."] = "ns"

    res_df = pd.DataFrame(results)
    res_df = res_df[["Parameter", "Type", "Prune Test", "p-value", "Prune Signif.", "Prune Trigger Risk"]].sort_values(by="p-value", ascending=True).reset_index(drop=True)
    print(res_df.to_string(index=False))


def run_statistical_importance_test(df: pd.DataFrame, cat_cols: list, num_cols: list):
    """Statistical importance tests with penalized rank imputation for pruned runs."""
    print("\n[ STATISTICAL PARAMETER IMPORTANCE & RECOMMENDATIONS ]")
    results = []

    eval_df = df[df["status"].isin(["COMPLETE", "PRUNED"])].copy()
    if len(eval_df) < 3:
        print("Not enough recorded trials to perform statistical significance tests.")
        return

    completed_df = eval_df[eval_df["status"] == "COMPLETE"]
    if len(completed_df) == 0:
        print("No completed trials found to infer baseline metrics.")
        return

    max_rmse = completed_df["mean_rmse"].max()
    penalty_val = max_rmse * 1.2
    
    eval_df["rank_score"] = eval_df["mean_rmse"].fillna(penalty_val)
    eval_df["log_rank_score"] = np.log10(eval_df["rank_score"])

    n_top = max(3, int(np.ceil(len(completed_df) * 0.25)))
    top_df = completed_df.sort_values(by="mean_rmse", ascending=True).head(n_top)
    best_trial = completed_df.sort_values(by="mean_rmse", ascending=True).iloc[0]

    for col in cat_cols:
        if col in eval_df.columns and eval_df[col].nunique() > 1:
            groups = [group["log_rank_score"].values for _, group in eval_df.groupby(col)]
            _, p_val = stats.kruskal(*groups) if len(groups) > 1 else (0.0, 1.0)
            
            top_choices = top_df[col].value_counts()
            rec_val = str(top_choices.index[0]) if not top_choices.empty else "N/A"

            results.append({
                "Parameter": col,
                "Type": "Categorical",
                "p-value": p_val,
                "Recommended (Top 25%)": rec_val,
                "Best Trial (#1)": str(best_trial[col])
            })

    for col in num_cols:
        if col in eval_df.columns and eval_df[col].nunique() > 1:
            sub_df = eval_df.dropna(subset=[col, "log_rank_score"])
            if len(sub_df) > 2:
                _, p_val = stats.spearmanr(sub_df[col], sub_df["log_rank_score"])
                
                min_v, max_v = top_df[col].min(), top_df[col].max()
                rec_str = format_num(min_v) if min_v == max_v else f"[{format_num(min_v)} - {format_num(max_v)}]"

                results.append({
                    "Parameter": col,
                    "Type": "Numeric",
                    "p-value": p_val,
                    "Recommended (Top 25%)": rec_str,
                    "Best Trial (#1)": format_num(best_trial[col])
                })

    if not results:
        print("No parameters varied sufficiently across recorded trials.")
        return

    p_values = [r["p-value"] for r in results if pd.notna(r["p-value"])]
    if p_values:
        q1 = np.percentile(p_values, 25)
        q2 = np.percentile(p_values, 50)
        q3 = np.percentile(p_values, 75)

        for r in results:
            p = r["p-value"]
            if pd.isna(p):
                r["Signif."] = "N/A"
            elif p <= q1:
                r["Signif."] = "***"
            elif p <= q2:
                r["Signif."] = "**"
            elif p <= q3:
                r["Signif."] = "*"
            else:
                r["Signif."] = "ns"

    res_df = pd.DataFrame(results)
    col_order = ["Parameter", "Type", "p-value", "Signif.", "Recommended (Top 25%)", "Best Trial (#1)"]
    res_df = res_df[col_order].sort_values(by="p-value", ascending=True).reset_index(drop=True)

    print(res_df.to_string(index=False))


def analyze_trials(csv_path: Path, top_k: int = 5, save: bool = False, output_dir: Path = Path("./output/optuna_logs")):
    """Executes full analysis and optionally logs output to file."""
    tee = None
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{csv_path.stem}_analysis.txt"
        tee = Tee(report_path)
        sys.stdout = tee

    try:
        df = load_and_clean_data(csv_path)

        print("=" * 80)
        print(f" OPTUNA LOG ANALYSIS: {csv_path.name}")
        print("=" * 80)

        completed_df = df[(df["status"] == "COMPLETE") & (df["mean_rmse"].notna())].copy()
        status_counts = df["status"].value_counts().to_dict()

        print("\n[ EXECUTION OVERVIEW ]")
        print(f"Total Trials Recorded : {len(df)}")
        print(f"Completed Trials     : {len(completed_df)}")
        print(f"Pruned Trials        : {status_counts.get('PRUNED', 0)}")

        if "duration_sec" in df.columns and df["duration_sec"].notna().any():
            durations = df["duration_sec"].dropna()
            print(f"Mean Trial Duration  : {format_time(durations.mean())}")
            print(f"Min Trial Duration   : {format_time(durations.min())}")
            print(f"Max Trial Duration   : {format_time(durations.max())}")

        if len(completed_df) == 0:
            print("\nNo completed trials found in log. Exiting analysis.")
            return

        # Print Top N Configurations
        print(f"\n[ TOP {top_k} BEST CONFIGURATIONS ]")
        top_trials = completed_df.sort_values(by="mean_rmse", ascending=True).head(top_k)
        
        candidate_cols = [
            "trial_num", "mean_rmse", "val_loss", "rmse_VAD",
            "criterion", "optimizer", "batch_size", "learning_rate",
            "weight_decay", "dropout_rate", "orth_loss_weight", "ccc_weight", "backbone_lr_scale",
            "VAD_weights_normalized"
        ]
        display_cols = [c for c in candidate_cols if c in top_trials.columns]
        
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", None)
        print(top_trials[display_cols].to_string(index=False))

        # Identify clean parameter lists for statistical tables
        cat_cols, num_cols = get_parameter_lists(df)

        print_categorical_table(df, cat_cols)
        run_statistical_importance_test(df, cat_cols, num_cols)
        run_prune_propensity_test(df, cat_cols, num_cols)
        run_joint_vad_analysis(df)

        if save and tee:
            print(f"\n[ REPORT SAVED ]\nAnalysis log written to: {output_dir / f'{csv_path.stem}_analysis.txt'}")

    finally:
        if tee:
            sys.stdout = tee.stdout
            tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Optuna Trial CSV Logs")
    parser.add_argument("--csv_path", type=str, nargs="?", default=None, help="Optional path to trial CSV file")
    parser.add_argument("--search_dir", type=str, default="./output", help="Directory to search if csv_path is not specified")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top trials to display")
    parser.add_argument("--save", action="store_true", help="Save text report of the analysis output")
    parser.add_argument("--output_dir", type=str, default="./output/optuna_logs", help="Directory where saved report files are stored")

    args = parser.parse_args()
    save_dir = Path(args.output_dir)

    if args.csv_path:
        target_path = Path(args.csv_path)
        if not target_path.exists():
            print(f"Error: File '{target_path}' not found.")
            exit(1)
        analyze_trials(target_path, top_k=args.top_k, save=args.save, output_dir=save_dir)
    else:
        search_path = Path(args.search_dir)
        discovered = discover_optuna_csvs(search_path)
        
        if not discovered:
            print(f"No Optuna trial CSVs found in '{search_path.resolve()}'.")
            exit(1)
            
        print(f"Discovered {len(discovered)} Optuna CSV log file(s). Analyzing the latest:")
        print(f" -> {discovered[0]}")
        analyze_trials(discovered[0], top_k=args.top_k, save=args.save, output_dir=save_dir)