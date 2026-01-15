import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob, os, re
from collections import defaultdict
from pathlib import Path

# ====== Settings ======
ROOT = r"C:\Users\ACER-PC\Desktop\WUIR-CLASS-recSys\UIR\results"  # cartella che contiene results_k*
BASELINE = 10.5                  # None per disattivare baseline
SAVEFIG = True
OUTDIR = os.path.join(ROOT, "saved_results")
os.makedirs(OUTDIR, exist_ok=True)

# dimensioni griglia per-modello
GRID_ROWS, GRID_COLS = 2, 5   # cambia a piacere
# ======================

# Pattern filename: <model>_k<k>_seed<seed>.txt
# Esempi validi: UIR-deficit-based_k2_seed44.txt, WUIR-strict-mastery_k3_seed45.txt
FN_RE = re.compile(r"(?P<model>.+?)_k(?P<k>\d+)_seed(?P<seed>\d+)\.txt$", re.IGNORECASE)

# ---- raccogliamo tutti i .txt dentro results_k*/seed* ----
paths = glob.glob(os.path.join(ROOT, "results_k5", "seed*", "*.txt"))
if not paths:
    raise FileNotFoundError(f"Nessun file .txt trovato sotto {ROOT}")

runs_by_group = defaultdict(list)  # key = (k, model) -> list of dfs (uno per seed)

for p in sorted(paths):
    fn = os.path.basename(p)
    m = FN_RE.match(fn)
    if not m:
        # salta file non conformi
        continue
    model = m.group("model")
    k = int(m.group("k"))
    seed = int(m.group("seed"))

    # carica singolo run
    df = pd.read_csv(p, sep=r"\s+", header=None, names=["step", "reward", "time"], on_bad_lines="skip")
    # assicurazioni minime
    df = df.dropna(subset=["step", "reward"])
    df = df.sort_values("step").reset_index(drop=True)
    df["k"] = k
    df["model"] = model
    df["seed"] = seed
    runs_by_group[(k, model)].append(df)

if not runs_by_group:
    raise RuntimeError("Pattern file non corrisponde a nessun .txt valido.")

# ---- aggregazione per (k, model) su più seed ----
agg_dfs = {}   # (k, model) -> df con step, reward_mean, reward_std, time_mean, n
global_xmin, global_xmax = np.inf, -np.inf
global_ymin, global_ymax = np.inf, -np.inf

for (k, model), run_list in runs_by_group.items():
    # unifichiamo l'asse step con merge outer + interpolazione
    # base: tutti gli step osservati in almeno un seed
    all_steps = pd.DataFrame({"step": sorted(set().union(*[set(r["step"].tolist()) for r in run_list]))})
    merged = all_steps.copy()

    # per reward
    for i, r in enumerate(run_list):
        merged = merged.merge(
            r[["step", "reward"]].rename(columns={"reward": f"reward_{i}"}),
            on="step", how="left"
        )

    # interpolazione lineare delle lacune (per step non presenti in un run)
    merged = merged.sort_values("step").reset_index(drop=True)
    reward_cols = [c for c in merged.columns if c.startswith("reward_")]
    merged[reward_cols] = merged[reward_cols].interpolate(method="linear", limit_direction="both")

    # statistiche
    merged["reward_mean"] = merged[reward_cols].mean(axis=1)
    merged["reward_std"]  = merged[reward_cols].std(axis=1, ddof=0)  # pop std; usa ddof=1 se preferisci sample std
    merged["n"] = len(run_list)

    # time medio (solo dove disponibile; non lo interpoliamo di default)
    time_frames = []
    for i, r in enumerate(run_list):
        t = r[["step", "time"]].rename(columns={"time": f"time_{i}"})
        time_frames.append(t)
    if time_frames:
        tm = time_frames[0]
        for t in time_frames[1:]:
            tm = tm.merge(t, on="step", how="outer")
        tm = tm.sort_values("step").reset_index(drop=True)
        tcols = [c for c in tm.columns if c.startswith("time_")]
        tm["time_mean"] = tm[tcols].mean(axis=1, skipna=True)
        merged = merged.merge(tm[["step", "time_mean"]], on="step", how="left")

    merged["model"] = model
    merged["k"] = k

    # aggiorna limiti globali per plot
    global_xmin = min(global_xmin, merged["step"].min())
    global_xmax = max(global_xmax, merged["step"].max())
    global_ymin = min(global_ymin, merged["reward_mean"].min())
    global_ymax = max(global_ymax, merged["reward_mean"].max())

    agg_dfs[(k, model)] = merged

# padding + baseline
pad_y = 0.03 * (global_ymax - global_ymin if global_ymax > global_ymin else 1.0)
if BASELINE is not None:
    global_ymin = min(global_ymin, float(BASELINE))
    global_ymax = max(global_ymax, float(BASELINE))
global_ymin -= pad_y
global_ymax += pad_y

# ---- plot per ogni k: tutti i modelli aggregati (media ± std) ----
for k in sorted({k for (k, _) in agg_dfs.keys()}):
    plt.figure(figsize=(9, 5))
    for (kk, model), df in agg_dfs.items():
        if kk != k:
            continue
        plt.plot(df["step"], df["reward_mean"], label=model, linewidth=1.5)
        plt.fill_between(df["step"],
                         df["reward_mean"] - df["reward_std"],
                         df["reward_mean"] + df["reward_std"],
                         alpha=0.18)

    if BASELINE is not None:
        plt.axhline(float(BASELINE), linestyle="--", linewidth=1.2, label="Baseline-5m-BEST")

    plt.title(f"Reward vs Step — Mean over seeds (k={k})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.xlim(global_xmin, global_xmax)
    plt.ylim(global_ymin, global_ymax)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(os.path.join(OUTDIR, f"combined_k{k}.png"), dpi=150)
    plt.show()

# ---- griglia per-modello (a gruppi dentro lo stesso k) ----
for k in sorted({k for (k, _) in agg_dfs.keys()}):
    subset = [(m, agg_dfs[(k, m)]) for (kk, m) in agg_dfs.keys() if kk == k]
    if not subset:
        continue

    rows, cols = GRID_ROWS, GRID_COLS
    n = len(subset)
    pages = int(np.ceil(n / (rows * cols)))
    for page in range(pages):
        fig, axes = plt.subplots(rows, cols, figsize=(14, 7), sharex=False, sharey=False)
        axes = axes.ravel()
        block = subset[page*(rows*cols):(page+1)*(rows*cols)]

        for ax, (model, df) in zip(axes, block):
            ax.plot(df["step"], df["reward_mean"], linewidth=1.4)
            ax.fill_between(df["step"],
                            df["reward_mean"] - df["reward_std"],
                            df["reward_mean"] + df["reward_std"],
                            alpha=0.18)
            if BASELINE is not None:
                ax.axhline(float(BASELINE), linestyle="--", linewidth=1.0)

            # best point + label del valore
            max_idx = df["reward_mean"].idxmax()
            x_max = df.loc[max_idx, "step"]
            y_max = df.loc[max_idx, "reward_mean"]

            # segna il punto
            ax.scatter(x_max, y_max, s=20, color="gold", edgecolor="black", zorder=5)

            # scrivi il valore accanto
            ax.text(
                x_max,
                y_max + 0.05,  # piccolo offset verso l'alto
                f"{y_max:.3f}",  # valore con 3 decimali
                fontsize=8,
                color="black",
                ha="center"
            )

            ax.set_title(model, fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_xlim(global_xmin, global_xmax)
            ax.set_ylim(global_ymin, global_ymax)
            ax.tick_params(labelsize=8)

        # spegni assi inutilizzati
        for j in range(len(block), rows*cols):
            axes[j].axis("off")

        fig.suptitle(f"Reward vs Step — Per Model (k={k})", fontsize=12)
        fig.supxlabel("Step", fontsize=10)
        fig.supylabel("Reward", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if SAVEFIG:
            fig.savefig(os.path.join(OUTDIR, f"grid_k{k}_p{page+1}.png"), dpi=150)
        plt.show()

# ---- summary per (k, model) sulla curva media ----
summary_rows = []
for (k, model), df in agg_dfs.items():
    max_idx = df["reward_mean"].idxmax()
    step_at_max = int(df.loc[max_idx, "step"])
    max_reward = float(df.loc[max_idx, "reward_mean"])

    std_at_max = float(df.loc[max_idx, "reward_std"])
    std_mean = float(df["reward_std"].mean())

    # tempo medio su tutto il run (se presente)
    time_mean = float(df["time_mean"].mean()) if "time_mean" in df else np.nan

    # avg ultime 50 valutazioni (sulla media)
    avg_last = float(df["reward_mean"].tail(50).mean())

    # primo step che supera (o eguaglia) baseline sulla media
    steps_to_baseline = None
    if BASELINE is not None:
        idxs = np.where(df["reward_mean"].values >= float(BASELINE))[0]
        if idxs.size > 0:
            steps_to_baseline = int(df.iloc[idxs[0]]["step"])

    summary_rows.append({
        "k": k,
        "Model": model,
        "Seeds": int(df["n"].iloc[0]),
        "Max Reward (mean)": max_reward,
        "Std at Max": std_at_max,
        "Mean Std (overall)": std_mean,
        "Step at Max": step_at_max,
        "Avg Reward (last 50, mean)": avg_last,
        #"Avg Time per eval (dataset)": time_mean,
        "Steps to baseline (mean)": steps_to_baseline
    })

df_summary = pd.DataFrame(summary_rows).sort_values(
    by=["k", "Max Reward (mean)"], ascending=[True, False]
).reset_index(drop=True)

print(df_summary)