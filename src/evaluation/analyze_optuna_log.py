import argparse
import ast
from pathlib import Path
import re
from io import StringIO
import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go


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
                    if isinstance(parsed, (list, tuple)):
                        arr = np.array(parsed, dtype=float)
                        total = arr.sum()
                        if total > 0:
                            norm_arr = arr / total
                            res = [round(float(x), 4) for x in norm_arr]
                            if len(res) == 2:
                                res.append(0.0)
                            return res
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


def determine_primary_metric(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Identifies the primary loss/metric column in order of priority:
    'rmse_overall', 'mean_rmse', 'val_loss'. Creates a unified '_primary_metric' column.
    """
    priority = ["rmse_overall", "mean_rmse", "val_loss"]
    chosen_metric = None
    for candidate in priority:
        if candidate in df.columns and df[candidate].dropna().count() > 0:
            chosen_metric = candidate
            break

    if chosen_metric is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        chosen_metric = numeric_cols[0] if len(numeric_cols) > 0 else "val_loss"

    df["_primary_metric"] = pd.to_numeric(df[chosen_metric], errors="coerce") if chosen_metric in df.columns else np.nan
    return df, chosen_metric



def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """Loads CSV log with automatic repair for unquoted bracketed lists and inconsistent field counts."""
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.ParserError, Exception):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()

            sanitized_lines = []
            for line in raw_lines:
                fixed = re.sub(r'(?<!")(\[[^\]]+\])(?!")', r'"\1"', line)
                sanitized_lines.append(fixed)

            df = pd.read_csv(
                StringIO("".join(sanitized_lines)),
                engine="python",
                on_bad_lines="skip"
            )
            print(f"Notice: Auto-repaired CSV formatting issues (unquoted lists/commas) in '{csv_path.name}'.")
        except Exception:
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
            print(f"Warning: Skipped malformed lines in '{csv_path.name}' due to CSV parsing errors.")

    df = parse_and_normalize_vad(df)

    categorical_cols_set = {"status", "VAD_weights_normalized", "criterion", "optimizer", "weights_source", "scheduler", "model"}
    for col in df.columns:
        if col not in categorical_cols_set and not col.startswith("_vad_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for v_col, a_col, d_col in [
        ("rmse_Valence", "rmse_Arousal", "rmse_Dominance"),
        ("rmse_Valence", "rmse_Arousal", "rmse_Dominance")
    ]:
        if v_col in df.columns and a_col in df.columns:
            if d_col in df.columns:
                df["rmse_VAD"] = df.apply(
                    lambda r: f"{format_num(r[v_col])}; {format_num(r[a_col])}; {format_num(r[d_col])}"
                    if pd.notna(r[v_col]) else "N/A",
                    axis=1
                )
            else:
                df["rmse_VAD"] = df.apply(
                    lambda r: f"{format_num(r[v_col])}; {format_num(r[a_col])}"
                    if pd.notna(r[v_col]) else "N/A",
                    axis=1
                )
            break

    df, primary_metric_name = determine_primary_metric(df)
    return df, primary_metric_name


def get_parameter_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Includes all parameter columns that vary across trials, excluding fixed or non-informative columns. Returns two lists: categorical and numeric parameters."""
    exclude_prefixes = ("_vad_", "mse_", "rmse_", "ccc_", "pred_std_", "datetime_")
    exclude = {
        "trial_num", "status", "duration_sec", "val_loss", "rmse_VAD",
        "_primary_metric", "rank_score", "log_rank_score", "is_pruned", 
        "weight_v", "weight_a", "weight_d", "VAD_weights_normalized"
    }

    target_numerics = [
        "batch_size", "orth_loss_weight", "ccc_weight", "head_lr", "backbone_lr", 
        "learning_rate", "weight_decay", "dropout_rate", "unfreeze_epoch", 
        "lr_factor", "lr_patience", "lr_warmup_epochs", "input_size"
    ]
    target_categoricals = ["model", "criterion", "optimizer", "input_size", "weights_source", "scheduler"]

    cat_cols = []
    num_cols = []

    eval_df = df[df["status"].isin(["COMPLETE", "PRUNED"])] if "status" in df.columns else df

    for col in df.columns:
        if col in exclude or any(col.startswith(prefix) for prefix in exclude_prefixes):
            continue

        valid_series = eval_df[col].dropna()
        if valid_series.nunique() <= 1:
            continue

        if col in target_categoricals:
            cat_cols.append(col)
        elif col in target_numerics or pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    return sorted(list(set(cat_cols))), sorted(list(set(num_cols)))


def format_num(val):
    """Utility helper for consistent number formatting in output tables."""
    if pd.isna(val) or val == "N/A":
        return "N/A"
    try:
        val = float(val)
    except (ValueError, TypeError):
        return str(val)

    if val.is_integer():
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


def extract_filename_params(csv_path: str) -> dict:
    """Extracts fixed hyperparameters encoded in the CSV log filename tag."""
    filename = Path(csv_path).name.replace("_trials.csv", "")
    params = {}

    boolean_flags = {
        "pretrained", "freezed", "data_augmentation", "use_amp", 
        "resume", "no_checkpoint", "no_model_save"
    }

    value_keys = [
        "seed", "model", "dataset", "input_size", "num_workers", "early_stopping_patience", 
        "output_dir", "batch_size", "epochs", "head_lr", "backbone_lr", "learning_rate",
        "unfreeze_epoch", "weight_decay", "weights_source", "dropout_rate", 
        "optimizer", "device", "grad_clip", "criterion", "VAD_weights", 
        "orth_loss_weight", "ccc_weight", "scheduler", "lr_factor", "lr_patience",
        "lr_warmup_epochs", "study_name", "pruner"
    ]

    for key in value_keys:
        match = re.search(rf"(?:^|_){key}-([^_]+)", filename)
        if match:
            val = match.group(1).replace("[", "\"").replace("]", "\"")
            params[key] = val

    for flag in boolean_flags:
        if re.search(rf"(?:^|_){flag}(?:_|$)", filename):
            params[flag] = True
        elif re.search(rf"(?:^|_)not{flag}(?:_|$)", filename):
            params[flag] = False

    return params


def generate_best_run_command(csv_path: str, best_row: pd.Series) -> str:
    """Constructs the exact executable CLI command for the best trial."""
    cmd_args = ["python", "-m", "src.training.train"]
    params = extract_filename_params(csv_path)
    
    known_args = [
        "dataset", "model", "input_size", "batch_size", "head_lr", "backbone_lr",
        "unfreeze_epoch", "weight_decay", "weights_source", 
        "dropout_rate", "optimizer", "criterion", "ccc_weight", 
        "VAD_weights", "orth_loss_weight", "lr_factor", "lr_patience",
        "lr_threshold", "lr_cooldown", "lr_min", "scheduler", "lr_warmup_epochs",
        "pretrained", "freezed", "data_augmentation", "use_amp", 
        "resume", "no_checkpoint", "no_model_save"
    ]
    
    for arg in known_args:
        if arg in best_row and pd.notna(best_row[arg]) and best_row[arg] != "N/A":
            params[arg] = best_row[arg]


    if "VAD_weights_normalized" in best_row and pd.notna(best_row["VAD_weights_normalized"]) and best_row["VAD_weights_normalized"] != "N/A":
        raw_vad = str(best_row["VAD_weights_normalized"]).strip("[]").replace(" ", "")
        params["VAD_weights"] = f'"{raw_vad}"'

    boolean_flags = {
        "pretrained", "freezed", "data_augmentation", "use_amp", 
        "resume", "no_checkpoint", "no_model_save"
    }

    for k, v in params.items():
        if k in boolean_flags:
            if v is True or str(v).lower() == "true":
                cmd_args.append(f"--{k}")
        else:
            cmd_args.append(f"--{k} {v}")

    return " ".join(cmd_args)


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
        top_df = completed_df.sort_values(by="_primary_metric", ascending=True).head(top_n)

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
            print(f"PERMANOVA Joint Normalized VAD Prune Risk Test p-value : {p_val:.4f}")


def print_categorical_table(df: pd.DataFrame, cat_cols: list):
    """Prints clear categorical summary table including prune rates."""
    if not cat_cols:
        return

    header  = "+-------------------+----------------+-------+-----------+-----------+-----------+------------+"
    divider = "+-------------------+----------------+-------+-----------+-----------+-----------+------------+"
    
    print("\n[ CATEGORICAL & DISCRETE PARAMETER SUMMARY ]")
    print(header)
    print(f"| {'Parameter':<17} | {'Category Value':<14} | {'Total':<5} | {'Mean Metric':<9} | {'Min Metric':<9} | {'Std Dev':<9} | {'Prune %':<10} |")
    print(header)

    for col in cat_cols:
        if col not in df.columns:
            continue
        
        completed = df[df["status"] == "COMPLETE"]
        group_comp = completed.groupby(col)["_primary_metric"].agg(["count", "mean", "min", "std"]).reset_index()
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
    """Identifies which parameter values trigger trial pruning."""
    if not cat_cols and not num_cols:
        return

    print("\n[ PRUNE PROPENSITY ANALYSIS (Parameters Triggering Pruning) ]")
    
    eval_df = df[df["status"].isin(["COMPLETE", "PRUNED"])].copy()
    if len(eval_df) < 5 or eval_df["status"].nunique() < 2:
        print("Requires both completed and pruned trials to compute prune propensity.")
        return

    eval_df["is_pruned"] = (eval_df["status"] == "PRUNED").astype(int)
    results = []

    for col in cat_cols:
        if col in eval_df.columns:
            sub_df = eval_df.dropna(subset=[col])
            if sub_df[col].nunique() > 1 and sub_df["is_pruned"].nunique() > 1:
                contingency = pd.crosstab(sub_df[col], sub_df["is_pruned"])
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    _, p_val, _, _ = stats.chi2_contingency(contingency)
                    
                    prune_rates = sub_df.groupby(col)["is_pruned"].mean()
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
        if col in eval_df.columns:
            sub_df = eval_df.dropna(subset=[col])
            pruned_vals = sub_df[sub_df["is_pruned"] == 1][col]
            completed_vals = sub_df[sub_df["is_pruned"] == 0][col]

            if len(pruned_vals) >= 1 and len(completed_vals) >= 1 and sub_df[col].nunique() > 1:
                _, p_val = stats.mannwhitneyu(pruned_vals, completed_vals, alternative="two-sided")
                
                med_pruned = pruned_vals.median()
                med_comp = completed_vals.median()
                direction = "High values" if med_pruned > med_comp else ("Low values" if med_pruned < med_comp else "Equal median")

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

    max_rmse = completed_df["_primary_metric"].max()
    penalty_val = max_rmse * 1.2
    
    eval_df["rank_score"] = eval_df["_primary_metric"].fillna(penalty_val)
    eval_df["log_rank_score"] = np.log10(eval_df["rank_score"])

    n_top = max(3, int(np.ceil(len(completed_df) * 0.25)))
    top_df = completed_df.sort_values(by="_primary_metric", ascending=True).head(n_top)
    best_trial = completed_df.sort_values(by="_primary_metric", ascending=True).iloc[0]

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


def analyze_trials(csv_path: Path, top_k: int = 5, save: bool = False, plot: bool = False, output_dir: Path = Path("./output/optuna_logs")):
    """Executes full analysis and optionally logs output to file."""
    tee = None
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{csv_path.stem}_analysis.txt"
        tee = Tee(report_path)
        sys.stdout = tee

    try:
        df, primary_metric_name = load_and_clean_data(csv_path)

        print("=" * 80)
        print(f" OPTUNA LOG ANALYSIS: {csv_path.name}")
        print(f" Primary Target Metric Identified: '{primary_metric_name}'")
        print("=" * 80)

        completed_df = df[(df["status"] == "COMPLETE") & (df["_primary_metric"].notna())].copy()
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

        if not completed_df.empty:
            best_row = completed_df.loc[completed_df["_primary_metric"].idxmin()]
            best_cmd = generate_best_run_command(csv_path, best_row)
            
            print(f"\n[ BEST TRIAL COMMAND LINE ]\n{best_cmd}")
        

        print(f"\n[ TOP {top_k} BEST CONFIGURATIONS ]")
        top_trials = completed_df.sort_values(by="_primary_metric", ascending=True).head(top_k)
        
        meta_cols = ["trial_num", "mean_rmse", "val_loss", "rmse_VAD", "duration_sec"]
        meta_present = [c for c in meta_cols if c in top_trials.columns]
        
        exclude_internal = {
            "trial_num", "status", "val_loss", "duration_sec", "rmse_VAD", "_primary_metric",
            "rmse_Valence", "rmse_Arousal", "rmse_Dominance",
            "ccc_Valence", "ccc_Arousal", "ccc_Dominance",
            "_vad_pV", "_vad_pA", "_vad_pD"
        }
        hp_present = [c for c in top_trials.columns if c not in exclude_internal]
        
        display_cols = meta_present + hp_present
        
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", None)
        print(top_trials[display_cols].to_string(index=False))

        cat_cols, num_cols = get_parameter_lists(df)

        print_categorical_table(df, cat_cols)
        run_statistical_importance_test(df, cat_cols, num_cols)
        run_prune_propensity_test(df, cat_cols, num_cols)
        if "_vad_pV" in df.columns and "_vad_pA" in df.columns and "_vad_pD" in df.columns:
            run_joint_vad_analysis(df)
        if plot:
            plot_interactive_ccc_vs_rmse(df, output_html=output_dir / f"{csv_path.stem}_ccc_vs_rmse_interactive.html")

    finally:
        if tee:
            sys.stdout = tee.stdout
            tee.close()



def plot_interactive_ccc_vs_rmse(df: pd.DataFrame, output_html: Path):
    """
    Generates an interactive scatter plot of CCC vs RMSE with hover tooltips
    showing all hyperparameters per trial. Saves output as an HTML file.
    """
    if 'mean_rmse' not in df.columns:
        rmse_cols = [c for c in df.columns if c.startswith("rmse_overall_")]
        if rmse_cols:
            df["mean_rmse"] = df[rmse_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        elif "rmse_overall" in df.columns:
            df["mean_rmse"] = pd.to_numeric(df["rmse_overall"], errors="coerce")

    if 'mean_ccc' not in df.columns:
        ccc_cols = [c for c in df.columns if c.startswith("ccc_overall_")]
        if ccc_cols:
            df["mean_ccc"] = df[ccc_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        elif "ccc_overall" in df.columns:
            df["mean_ccc"] = pd.to_numeric(df["ccc_overall"], errors="coerce")

    valid_df = df.dropna(subset=['mean_rmse', 'mean_ccc']).copy()
    if valid_df.empty:
        print("⚠️ No valid numeric data found for plotting.")
        return

    # Filter out non-hyperparameter columns for hover tooltips
    exclude_cols = {
        "_primary_metric", "_vad_pV", "_vad_pA", "_vad_pD", "mean_rmse", "mean_ccc", "trial_num", "status", "duration_sec", "val_loss", "rmse_VAD"
    }
    hp_cols = [
        c for c in valid_df.columns 
        if not c.startswith(("mse_", "rmse_", "ccc_", "pred_std_", "_")) and c not in exclude_cols
    ]

    fig = go.Figure()

    status_config = {
        'COMPLETE': {'color': '#2ca02c', 'symbol': 'circle', 'name': 'COMPLETE'},
        'PRUNED': {'color': '#ff7f0e', 'symbol': 'triangle-up', 'name': 'PRUNED'}
    }

    for status, cfg in status_config.items():
        sub_df = valid_df[valid_df['status'].astype(str).str.upper() == status]
        if sub_df.empty:
            continue

        hover_text_list = []
        for _, row in sub_df.iterrows():
            trial_id = int(row['trial_num']) if 'trial_num' in row and pd.notna(row['trial_num']) else row.name
            
            # Format hover tooltip HTML
            card = [
                f"<b>Trial #{trial_id}</b> ({status})",
                f"<b>Overall RMSE:</b> {row['mean_rmse']:.4f}",
                f"<b>Overall CCC:</b> {row['mean_ccc']:.4f}",
                f"<b>Duration:</b> {row['duration_sec']:.2f} seconds"
            ]
            if 'val_loss' in row and pd.notna(row['val_loss']) and str(row['val_loss']) != 'N/A':
                card.append(f"<b>Val Loss:</b> {row['val_loss']}")
            
            card.append("")
            card.append("<b>Hyperparameters:</b>")
            for hp in hp_cols:
                val = row[hp]
                if pd.notna(val) and str(val) != 'N/A':
                    card.append(f"• <i>{hp}:</i> {val}")

            hover_text_list.append("<br>".join(card))

        fig.add_trace(go.Scatter(
            x=sub_df['mean_rmse'],
            y=sub_df['mean_ccc'],
            mode='markers+text',
            name=cfg['name'],
            text=[f"T{int(r['trial_num'])}" if 'trial_num' in r and pd.notna(r['trial_num']) else f"T{r.name}" for _, r in sub_df.iterrows()],
            textposition="top center",
            textfont=dict(size=10),
            hoverinfo='text',
            hovertext=hover_text_list,
            marker=dict(
                size=12,
                color=cfg['color'],
                symbol=cfg['symbol'],
                line=dict(width=1, color='black'),
                opacity=0.85
            )
        ))

    fig.update_layout(
        title=dict(
            text="Interactive Optuna Trial Evaluation: CCC vs. RMSE",
            font=dict(size=16, color="black")
        ),
        xaxis=dict(title="Overall RMSE (Lower is better)", gridcolor='#E5E5E5'),
        yaxis=dict(title="Overall CCC (Higher is better)", gridcolor='#E5E5E5'),
        template="plotly_white",
        legend=dict(title="Trial Status", bordercolor="black", borderwidth=1),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )

    print("\n[ SAVING INTERACTIVE CCC vs RMSE PLOT ]")
    if output_html:
        fig.write_html(output_html)
        print(f"Interactive HTML report saved to '{output_html}'")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Optuna Trial CSV Logs")
    parser.add_argument("--csv_path", type=str, nargs="?", default=None, help="Optional path to trial CSV file")
    parser.add_argument("--search_dir", type=str, default="./output", help="Directory to search if csv_path is not specified")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top trials to display")
    parser.add_argument("--save", action="store_true", help="Save text report of the analysis output")
    parser.add_argument("--plot", action="store_true", help="Generate interactive CCC vs RMSE plot and save as HTML")
    parser.add_argument("--output_dir", type=str, default="./output/optuna_logs", help="Directory where saved report files are stored")

    args = parser.parse_args()
    save_dir = Path(args.output_dir)

    if args.csv_path:
        target_path = Path(args.csv_path)
        if not target_path.exists():
            print(f"Error: File '{target_path}' not found.")
            exit(1)
        analyze_trials(target_path, top_k=args.top_k, save=args.save, plot=args.plot, output_dir=save_dir)
    else:
        search_path = Path(args.search_dir)
        discovered = discover_optuna_csvs(search_path)
        
        if not discovered:
            print(f"No Optuna trial CSVs found in '{search_path.resolve()}'.")
            exit(1)
            
        print(f"Discovered {len(discovered)} Optuna CSV log file(s). Analyzing the latest:")
        print(f" -> {discovered[0]}")
        for study in discovered:
            analyze_trials(study, top_k=args.top_k, save=args.save, plot=args.plot, output_dir=save_dir)