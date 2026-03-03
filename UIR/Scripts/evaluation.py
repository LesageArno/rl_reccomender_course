import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd



# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PlotConfig:
    # ====== Settings ======
    ROOT: str = str((Path(__file__).resolve().parent / ".." / "results").resolve())
    BASELINE: Optional[float] = None                  # None to deactivate baseline       [2.0, 4.1, 8.0, 10.5] #values for frej comparison BEST POSSIBLE
    SAVEFIG: bool = True
    OUTDIR_NAME: str = "saved_results_preferences_v3"

    # Folder under ROOT where your experiment results are stored
    # Example: preferences5
    EXPERIMENT_SUBDIR: str = "preferences_v3_k3"

    GRID_ROWS: int = 2
    GRID_COLS: int = 4

    # Which models to include (simple allowlist)
    ALLOWED_MODELS: Tuple[str, ...] = ("Employability", "UIR-threshold-based", "UIR-gap-based")

    # Metrics available in new logs
    # New logs: step E_goal reward gap pref time
    METRICS: Tuple[str, ...] = ("E_goal", "gap", "pref", "skills_covered", "skills_missing")  # add "reward" if you also want to plot/debug it

    # Whether to run or not the statistical test
    RUN_STAT_TESTS: bool = True
    DEBUG_STAT: bool = True


@dataclass
class MetricSpec:
    ylabel: str
    best: str  # "max" or "min"
    baseline: Optional[float] = None


def build_metric_specs(cfg: PlotConfig) -> Dict[str, MetricSpec]:
    """
    Define how each metric should be displayed and how to compute "best".
    - E_goal: higher is better
    - gap: lower is better
    - pref: higher is better
    """
    specs = {
        "E_goal": MetricSpec(ylabel="E_goal", best="max", baseline=cfg.BASELINE),
        "gap": MetricSpec(ylabel="Skill gap", best="min", baseline=None),
        "pref": MetricSpec(ylabel="Preference coverage", best="max", baseline=None),
        "reward": MetricSpec(ylabel="Reward (debug)", best="max", baseline=None),
        #"req_levels": MetricSpec(ylabel="Req levels", best="max", baseline=None),
        #"req_skills": MetricSpec(ylabel="Req skills", best="max", baseline=None),
        "skills_covered": MetricSpec(ylabel="Skills covered", best="max", baseline=None),
        "skills_missing": MetricSpec(ylabel="Skills missing", best="min", baseline=None),
    }
    # Return only those the user enabled in cfg.METRICS
    out = {}
    for m in cfg.METRICS:
        if m not in specs:
            raise ValueError(f"Metric '{m}' is not recognized. Add it to build_metric_specs().")
        out[m] = specs[m]
    return out


# =============================================================================
# File discovery + parsing
# =============================================================================

# Pattern filename: <model>_k<k>_seed<seed>.txt
FN_RE = re.compile(r"(?P<model>.+?)_k(?P<k>\d+)_seed(?P<seed>\d+)\.txt$", re.IGNORECASE)


def find_result_files(cfg: PlotConfig) -> List[str]:
    """
    Discover all result files under:
        ROOT / EXPERIMENT_SUBDIR / seed* / *.txt
    """
    root = cfg.ROOT
    exp = cfg.EXPERIMENT_SUBDIR
    pattern = os.path.join(root, exp, "seed*", "*.txt")
    paths = glob.glob(pattern)
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No .txt found under: {pattern}")
    return paths


def parse_run_metadata(filepath: str) -> Optional[Tuple[str, int, int]]:
    """
    Parse filename metadata: (model, k, seed).
    Returns None if filename does not match the expected pattern.
    """
    fn = os.path.basename(filepath)
    m = FN_RE.match(fn)
    if not m:
        return None
    model = m.group("model")
    k = int(m.group("k"))
    seed = int(m.group("seed"))
    return model, k, seed


def load_single_run(filepath: str, model: str, k: int, seed: int) -> pd.DataFrame:
    """
    Load a single run file into a DataFrame.

    # Supported formats:
    # - 10 columns (latest): step, E_goal, reward, gap, pref, req_levels, req_skills, skills_covered, skills_missing, time
    # - 6 columns  (new):    step, E_goal, reward, gap, pref, time
    # - 3 columns  (legacy): step, reward, time

    Returns a DataFrame with at least:
    - step
    - E_goal, gap, pref, reward, req_levels, req_skills, skills_covered, skills_missing (possibly NaN if old format)
    - time (if present)
    - model, k, seed
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None)

    cols_last = ["step", "E_goal", "reward", "gap", "pref","req_levels", "req_skills", "skills_covered", "skills_missing" ,"time"]

    # New logs
    cols_new = ["step", "E_goal", "reward", "gap", "pref", "time"]
    # Old logs (retro-compat)
    cols_old = ["step", "reward", "time"]

    if len(df.columns) == len(cols_last):
        df.columns = cols_last
    elif len(df.columns) == 6:
        df.columns = cols_new
        # Fill missing new-metrics columns to keep code uniform
        df["req_levels"] = np.nan
        df["req_skills"] = np.nan
        df["skills_covered"] = np.nan
        df["skills_missing"] = np.nan
    elif len(df.columns) == 3:
        df.columns = cols_old
        # Fill missing new-metrics columns to keep code uniform
        df["E_goal"] = np.nan
        df["gap"] = np.nan
        df["pref"] = np.nan
        df["req_levels"] = np.nan
        df["req_skills"] = np.nan
        df["skills_covered"] = np.nan
        df["skills_missing"] = np.nan
    else:
        raise ValueError(f"Unknown log format: {filepath} has {len(df.columns)} columns")

    # Basic cleaning
    df = df.dropna(subset=["step"])
    df = df.sort_values("step").reset_index(drop=True)

    # Attach metadata
    df["model"] = model
    df["k"] = k
    df["seed"] = seed

    return df


def load_all_runs(cfg: PlotConfig) -> Dict[Tuple[int, str], List[pd.DataFrame]]:
    """
    Load all runs and group them by (k, model).

    Returns:
        runs_by_group[(k, model)] = [df_seed1, df_seed2, ...]
    """
    paths = find_result_files(cfg)

    runs_by_group: Dict[Tuple[int, str], List[pd.DataFrame]] = {}

    for p in paths:
        meta = parse_run_metadata(p)
        if meta is None:
            continue

        model, k, seed = meta

        # Keep only the models you care about
        if cfg.ALLOWED_MODELS and model not in cfg.ALLOWED_MODELS:
            continue

        df = load_single_run(p, model=model, k=k, seed=seed)

        key = (k, model)
        if key not in runs_by_group:
            runs_by_group[key] = []
        runs_by_group[key].append(df)

    if not runs_by_group:
        raise RuntimeError("No valid runs found. Check filenames and ALLOWED_MODELS filter.")

    return runs_by_group


# =============================================================================
# Aggregation (mean/std across seeds) per (k, model)
# =============================================================================

def build_union_step_axis(run_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Build a DataFrame containing the union of all 'step' values observed across runs.
    This keeps aggregation stable even if seeds have slightly different evaluation steps.
    """
    all_steps_set = set()
    for r in run_list:
        for s in r["step"].tolist():
            all_steps_set.add(int(s))
    all_steps = sorted(all_steps_set)
    return pd.DataFrame({"step": all_steps})


def merge_metric_columns(base: pd.DataFrame, run_list: List[pd.DataFrame], metric: str) -> pd.DataFrame:
    """
    Merge one metric across seeds into columns: metric_0, metric_1, ...
    """
    merged = base.copy()

    for i, r in enumerate(run_list):
        if metric not in r.columns:
            continue
        tmp = r[["step", metric]].rename(columns={metric: f"{metric}_{i}"})
        merged = merged.merge(tmp, on="step", how="left")

    return merged


def interpolate_metric_columns(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Interpolate missing points for a metric over steps to allow mean/std across seeds.
    """
    out = df.copy()
    cols = []
    for c in out.columns:
        if c.startswith(f"{metric}_"):
            cols.append(c)

    if not cols:
        out[f"{metric}_mean"] = np.nan
        out[f"{metric}_std"] = np.nan
        return out

    out = out.sort_values("step").reset_index(drop=True)
    out[cols] = out[cols].interpolate(method="linear", limit_direction="both")

    out[f"{metric}_mean"] = out[cols].mean(axis=1)
    out[f"{metric}_std"] = out[cols].std(axis=1, ddof=0)  

    return out


def aggregate_group(run_list: List[pd.DataFrame], metrics: List[str]) -> pd.DataFrame:
    """
    Aggregate a list of runs (different seeds) into a single DataFrame:
    - step axis = union of steps across seeds
    - for each metric: metric_mean, metric_std
    - n = number of seeds
    - time_mean if time exists
    """
    step_axis = build_union_step_axis(run_list)
    merged = step_axis.copy()

    # Merge and compute mean/std for each metric
    for metric in metrics:
        merged = merge_metric_columns(merged, run_list, metric)
        merged = interpolate_metric_columns(merged, metric)

    # Attach seed count
    merged["n"] = len(run_list)

    # Time mean (no interpolation by default)
    if "time" in run_list[0].columns:
        # Merge time across seeds into time_0, time_1, ...
        tm = step_axis.copy()
        for i, r in enumerate(run_list):
            if "time" not in r.columns:
                continue
            tmp = r[["step", "time"]].rename(columns={"time": f"time_{i}"})
            tm = tm.merge(tmp, on="step", how="left")
        time_cols = [c for c in tm.columns if c.startswith("time_")]
        if time_cols:
            tm["time_mean"] = tm[time_cols].mean(axis=1, skipna=True)
            merged = merged.merge(tm[["step", "time_mean"]], on="step", how="left")

    return merged


def aggregate_all_groups(runs_by_group: Dict[Tuple[int, str], List[pd.DataFrame]],
                         cfg: PlotConfig) -> Dict[Tuple[int, str], pd.DataFrame]:
    """
    Aggregate all (k, model) groups.
    """
    metrics = list(cfg.METRICS)
    agg_dfs: Dict[Tuple[int, str], pd.DataFrame] = {}

    for (k, model), run_list in runs_by_group.items():
        agg = aggregate_group(run_list, metrics)
        agg["k"] = k
        agg["model"] = model
        agg_dfs[(k, model)] = agg

    return agg_dfs


# =============================================================================
# Plotting
# =============================================================================

def compute_ylim_for_metric(agg_dfs: Dict[Tuple[int, str], pd.DataFrame],
                            k: int,
                            metric: str,
                            baseline: Optional[float]) -> Optional[Tuple[float, float]]:
    """
    Compute y-limits (min/max) for a given metric and k across all models.
    """
    ymin = np.inf
    ymax = -np.inf

    for (kk, _model), df in agg_dfs.items():
        if kk != k:
            continue
        col = f"{metric}_mean"
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        if np.all(np.isnan(y)):
            continue
        ymin = min(ymin, float(np.nanmin(y)))
        ymax = max(ymax, float(np.nanmax(y)))

    if ymin == np.inf:
        return None

    if baseline is not None:
        ymin = min(ymin, float(baseline))
        ymax = max(ymax, float(baseline))

    pad = 0.03 * (ymax - ymin if ymax > ymin else 1.0)
    return (ymin - pad, ymax + pad)

def plot_overview_for_k(agg_dfs: Dict[Tuple[int, str], pd.DataFrame],
                        metric_specs: Dict[str, MetricSpec],
                        cfg: PlotConfig,
                        outdir: str,
                        k: int) -> None:
    """
    1 figure per k with nb_metrics subplots (E_goal, gap, pref, ...).
    Each subplot contains: model curves (mean ± std).
    """
    metrics = list(metric_specs.keys())
    n = len(metrics)

    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        spec = metric_specs[metric]
        ylims = compute_ylim_for_metric(agg_dfs, k, metric, spec.baseline)
        if ylims is None:
            ax.set_visible(False)
            continue

        for (kk, model), df in agg_dfs.items():
            if kk != k:
                continue

            y = df[f"{metric}_mean"]
            ystd = df[f"{metric}_std"]
            if y.isna().all():
                continue

            ax.plot(df["step"], y, linewidth=1.5, label=model)
            ax.fill_between(df["step"], y - ystd, y + ystd, alpha=0.18)

        if spec.baseline is not None:
            ax.axhline(float(spec.baseline), linestyle="--", linewidth=1.0, label="Baseline")

        ax.set_title(metric)
        ax.set_xlabel("Step")
        ax.set_ylabel(spec.ylabel)
        ax.set_ylim(*ylims)
        ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Overview — metrics vs step (k={k})", fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    if cfg.SAVEFIG:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"overview_k{k}.png"), dpi=150)

    plt.show()

def plot_metric_models_for_k(agg_dfs: Dict[Tuple[int, str], pd.DataFrame],
                             metric: str,
                             spec: MetricSpec,
                             cfg: PlotConfig,
                             outdir: str,
                             k: int) -> None:
    """
    1 figure per (k, metric) with subplots (one per model).
    Each subplot: model with marker on best reached point(max/min).
    """
    # prendi i modelli disponibili per quel k
    models = sorted([m for (kk, m) in agg_dfs.keys() if kk == k])
    if not models:
        return

    ylims = compute_ylim_for_metric(agg_dfs, k, metric, spec.baseline)
    if ylims is None:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4), sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        df = agg_dfs[(k, model)]
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        y = df[mean_col]
        ystd = df[std_col]

        ax.plot(df["step"], y, linewidth=1.6)
        ax.fill_between(df["step"], y - ystd, y + ystd, alpha=0.18)

        # best point (max/min)
        if spec.best == "max":
            best_row = df.loc[y.idxmax()]
        elif spec.best == "min":
            best_row = df.loc[y.idxmin()]
        else:
            raise ValueError("MetricSpec.best must be 'max' or 'min'")

        x_best = best_row["step"]
        y_best = best_row[mean_col]

        ax.scatter([x_best], [y_best], s=45, zorder=5)
        ax.text(x_best, y_best, f"  {y_best:.3f}", fontsize=9, va="bottom")

        if spec.baseline is not None:
            ax.axhline(float(spec.baseline), linestyle="--", linewidth=1.0)

        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Step")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(*ylims)

    axes[0].set_ylabel(spec.ylabel)
    fig.suptitle(f"{metric} — per model (k={k})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if cfg.SAVEFIG:
        metric_dir = os.path.join(outdir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        fig.savefig(os.path.join(metric_dir, f"models_{metric}_k{k}.png"), dpi=150)

    plt.show()



def plot_metric_for_k(agg_dfs: Dict[Tuple[int, str], pd.DataFrame],
                      k: int,
                      metric: str,
                      spec: MetricSpec,
                      cfg: PlotConfig,
                      outdir: str) -> None:
    """
    Plot one metric vs step for a fixed k, with mean ± std across seeds.
    """
    ylims = compute_ylim_for_metric(agg_dfs, k, metric, spec.baseline)

    if ylims is None:
        return

    plt.figure(figsize=(9, 5))

    for (kk, model), df in agg_dfs.items():
        if kk != k:
            continue

        y = df[f"{metric}_mean"]
        ystd = df[f"{metric}_std"]

        if y.isna().all():
            continue

        plt.plot(df["step"], y, label=model, linewidth=1.5)
        plt.fill_between(df["step"], y - ystd, y + ystd, alpha=0.18)

    # Baseline line (only for metrics where it makes sense)
    if spec.baseline is not None:
        plt.axhline(float(spec.baseline), linestyle="--", linewidth=1.2, label="Baseline")

    plt.title(f"{metric} vs Step — Mean over seeds (k={k})")
    plt.xlabel("Step")
    plt.ylabel(spec.ylabel)
    plt.ylim(*ylims)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    if cfg.SAVEFIG:
        metric_dir = os.path.join(outdir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        plt.savefig(os.path.join(metric_dir, f"combined_{metric}_k{k}.png"), dpi=150)

    plt.show()



def plot_all(agg_dfs: Dict[Tuple[int, str], pd.DataFrame],
             metric_specs: Dict[str, MetricSpec],
             cfg: PlotConfig,
             outdir: str) -> None:
    """
    For each metric and each k, generate a combined plot across models.
    """
    ks = sorted({k for (k, _m) in agg_dfs.keys()})

    for metric, spec in metric_specs.items():
        for k in ks:
            plot_metric_for_k(agg_dfs, k, metric, spec, cfg, outdir)


# =============================================================================
# Summary table
# =============================================================================

def pick_best_index(series: pd.Series, best: str) -> int:
    """
    Return index of best point.
    best == "max": argmax
    best == "min": argmin
    """
    if best == "max":
        return int(series.idxmax())
    if best == "min":
        return int(series.idxmin())
    raise ValueError("best must be 'max' or 'min'")


def build_summary(agg_dfs: Dict[Tuple[int, str], pd.DataFrame],
                  metric_specs: Dict[str, MetricSpec],
                  cfg: PlotConfig,
                  runs_by_group: Dict[Tuple[int, str], List[pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Build one summary table with:
    - k, model, metric
    - best mean, std at best, step at best
    - avg last 50 evaluations
    - seeds
    """
    rows = []

    stat_results = {}

    if cfg.RUN_STAT_TESTS and runs_by_group is not None:            #PERFORMING TUKEY HDS (STATISTICAL TEST)
        ks = sorted({k for (k, _m) in agg_dfs.keys()})
        for metric, spec in metric_specs.items():
            for k in ks:
                group_data = []
                model_names = []
                # for each metric and for each k we found the best value
                for (kk, model), df_list in runs_by_group.items():
                    if kk == k:
                        # Best value for each seed
                        seeds_best = [ (df_s[metric].max() if spec.best=="max" else df_s[metric].min()) for df_s in df_list ]
                        group_data.append(seeds_best)
                        model_names.append(model)
                
                if len(group_data) > 1:
                    try:
                        # Mean computation
                        means = [np.mean(g) for g in group_data]
                        best_idx = np.argmax(means) if spec.best == "max" else np.argmin(means)
                        best_model = model_names[best_idx]
                        
                        flattened_data = [item for sublist in group_data for item in sublist]
                        labels = [m for i, m in enumerate(model_names) for _ in range(len(group_data[i]))]
                        res = pairwise_tukeyhsd(endog=flattened_data, groups=labels, alpha=0.05)

                        if cfg.DEBUG_STAT:
                            print(f"\n>>> DEBUG STAT: {metric} (k={k})")
                            for name, val in zip(model_names, group_data):
                                print(f"  Model: {name:20} | Seed Values: {np.round(val,2)} | Mean: {np.mean(val):.2f}")
                            print(res.summary()) # Print Tukey table with meandiff and p-adj
                        
                        # Interpretation: who is similar to best?
                        stat_results[(k, metric, best_model)] = "BEST"
                        for _, row in pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0]).iterrows():
                            if row['group1'] == best_model and not row['reject']: stat_results[(k, metric, row['group2'])] = "Similar"
                            if row['group2'] == best_model and not row['reject']: stat_results[(k, metric, row['group1'])] = "Similar"
                    except: pass

    for (k, model), df in agg_dfs.items():
        seeds = int(df["n"].iloc[0]) if "n" in df.columns else 1

        for metric, spec in metric_specs.items():
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col not in df.columns:
                continue

            m = df[mean_col]
            if m.isna().all():
                continue

            idx = pick_best_index(m, spec.best)

            stat_flag = stat_results.get((k, metric, model), "Inferior" if cfg.RUN_STAT_TESTS else "N/A")

            rows.append({
                "k": k,
                "Model": model,
                "Metric": metric,
                "Seeds": seeds,
                "Best (mean)": float(df.loc[idx, mean_col]),
                "Std at Best": float(df.loc[idx, std_col]) if std_col in df.columns else np.nan,
                "Step at Best": int(df.loc[idx, "step"]),
                "Avg last 50 (mean)": float(m.tail(50).mean()),
                "Statistical Sig.": stat_flag,
            })

    out = pd.DataFrame(rows)

    # Sorting: for gap we want ascending, for others descending
    def sort_key(row):
        metric = row["Metric"]
        val = row["Best (mean)"]
        if metric == "gap":
            return (row["k"], row["Model"], val)     # smaller better
        else:
            return (row["k"], row["Model"], -val)    # larger better

    if not out.empty:
        out = out.sort_values(by=["Metric", "k", "Model", "Best (mean)"], ascending=True).reset_index(drop=True)

    return out


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = PlotConfig()

    outdir = os.path.join(cfg.ROOT, cfg.OUTDIR_NAME)
    os.makedirs(outdir, exist_ok=True)

    metric_specs = build_metric_specs(cfg)

    # 1) Load all runs grouped by (k, model)
    runs_by_group = load_all_runs(cfg)

    # 2) Aggregate runs across seeds (mean/std)
    agg_dfs = aggregate_all_groups(runs_by_group, cfg)

    # 3) Plot (combined plots: each metric × each k)
    #plot_all(agg_dfs, metric_specs, cfg, outdir)
    ks = sorted({k for (k, _m) in agg_dfs.keys()})

    # (1) 1 figura per k con 3 subplots (metriche)
    for k in ks:
        plot_overview_for_k(agg_dfs, metric_specs, cfg, outdir, k)

    # (2) 3 figure per k (una per metrica), con 3 subplots (modelli) e best point
    for k in ks:
        for metric, spec in metric_specs.items():
            plot_metric_models_for_k(agg_dfs, metric, spec, cfg, outdir, k)


    # 4) Summary table
    df_summary = build_summary(agg_dfs, metric_specs, cfg, runs_by_group)
    print(df_summary)

    if cfg.SAVEFIG and not df_summary.empty:
        df_summary.to_csv(os.path.join(outdir, "summary_all_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
