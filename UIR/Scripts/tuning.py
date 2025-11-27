import copy
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.trial import TrialState
from collections import deque
import optuna.visualization as ov

from Dataset import Dataset
from Reinforce import Reinforce
import json
import argparse
import yaml
import numpy as np

import torch
torch.distributions.Distribution.set_default_validate_args(False)

class ASHAReinforceTuner:
    """
    Tuning iperparametri per Reinforce con Optuna + ASHA (Successive Halving).
    """

    def __init__(self, base_config, total_steps=200_000, eval_freq=5_000,
                 n_trials=30, grace_period=50_000, reduction_factor=3, n_jobs=4):
        self.base_config = copy.deepcopy(base_config)
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.n_trials = n_trials
        self.seed = self.base_config.get('seed', 42)
        self.min_resource = max(1, grace_period // eval_freq)
        self.reduction_factor = reduction_factor
        self.n_jobs = n_jobs

    # ---------------- TUNING PRINCIPALE ---------------- #
    def tune(self):
        '''pruner = SuccessiveHalvingPruner(
            min_resource=self.min_resource,
            reduction_factor=self.reduction_factor,
            bootstrap_count=0 #2 * self.n_jobs,  # Per stabilizzare il pruning iniziale
        )'''

        pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            storage="sqlite:///asha_reinforce.db",  
            study_name="1000courses_k5_1000jobs_35skillscurricuum",
            load_if_exists=True
        )

        n_complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
        n_restart = max(0, self.n_trials - n_complete)

        print("%=================================================================%")
        print(n_restart, "trials da eseguire.")
        print("%=================================================================%")

        study.sampler=optuna.samplers.TPESampler(seed=None, #self.seed,
                                                n_startup_trials=5,
                                                multivariate=True,
                                                group=True
                                                )
        
        print(n_restart, "trials da eseguire.")

        if n_restart > 0:
            study.optimize(self._objective, n_trials=n_restart, n_jobs=self.n_jobs)
        else:
            print("target trial reached.")
            return study.best_trial.params
        fig = ov.plot_optimization_history(study)
        fig.write_html("optimization_history1000.html")
        fig.show()
        return study.best_trial.params # self._sample_weights(study.best_trial)#self._sample_hparams(study.best_trial)

    # ---------------- FUNZIONE OBIETTIVO ---------------- #
    def _objective(self, trial):
        # 1) campionamento iperparametri
        # hparams = self._sample_hparams(trial)
        weights = self._sample_weights(trial)
        n_repeats = 3
        global_step = [0] 

        run_scores = []
        for r in range(n_repeats):
            
            # 2) prepara config e dataset
            cfg = copy.deepcopy(self.base_config)
            cfg["total_steps"] = self.total_steps
            cfg["eval_freq"] = self.eval_freq
            cfg["seed"] = cfg.get("seed", self.seed) + r # + trial.number
            

            dataset = Dataset(cfg)
            print(dataset)

            # 3) istanzia Reinforce con override_hparams
            model = Reinforce(
                dataset=dataset,
                model=cfg["model"],
                k=cfg["k"],
                threshold=cfg["threshold"],
                run=0,
                save_name=f"{cfg['name_exp']}_trial{trial.number}",
                total_steps=self.total_steps,
                eval_freq=self.eval_freq,
                feature=cfg["feature"],
                baseline=cfg["baseline"],
                method=cfg["method"],
                beta1=weights["beta1"],
                beta2=1.0,  # weights["beta2"],
                params=self.base_config["hypers"] #hparams
            )

            ctx = {
                    "history": [],
                    "best": float("-inf"),
                    "reports": global_step[0],
                    "to_prune": False,
                    "hist_raw": deque(maxlen=6),
                    "confirm_streak": 0,
                    "since_best_reports": 0,
                    "cooldown": 0,
                    "WARMUP_EVALS": max(3, self.min_resource),
                    "ES_DELTA": 0.03,
                    "CONFIRM_K": 2,
                    "BASE_PATIENCE": 8,
                    "HIGH_VAR_PATIENCE": 12,
                    "VAR_THRESHOLD": 0.05,
                    "COOLDOWN_AFTER_BEST": 2,
                }

            if hasattr(model, "eval_callback"):
                model.eval_callback.report_fn = lambda step, avg: (
                    ctx["history"].append(float(avg)),
                    self.report_fn(step, avg, trial, weights, ctx, global_step)
                    )[-1]

            # 5) avvia training
            model.reinforce_recommendation()

            # 6) leggi la metrica finale dal callback
            history = ctx["history"]
            avg = np.mean(history[-10:])
            #avg = getattr(model.eval_callback, "last_avg_jobs", None)
            if avg is None:
                raise optuna.TrialPruned()
            
            run_scores.append(avg)
            
            '''if getattr(model.eval_callback, "was_pruned", False):
                to_prune = True

            if to_prune:
                raise optuna.TrialPruned()'''
            
        robust_score = float(np.quantile(run_scores, 0.25))

        return robust_score
    

    def report_fn(self, step, avg, trial, weights, ctx, global_step):
        current = float(avg)
        ctx["hist_raw"].append(current)

        # std breve
        if len(ctx["hist_raw"]) >= 2:
            m = sum(ctx["hist_raw"]) / len(ctx["hist_raw"])
            std = (sum((x - m) ** 2 for x in ctx["hist_raw"]) / len(ctx["hist_raw"])) ** 0.5
        else:
            std = 0.0
        patience = ctx["HIGH_VAR_PATIENCE"] if std > ctx["VAR_THRESHOLD"] else ctx["BASE_PATIENCE"]

        # report Optuna
        trial.report(current, step=global_step[0])
        global_step[0] += 1
        return True
    
# ---------------- UPDATE GLOBAL BEST ---------------- #

    def update_study_best(self, step, value, trial, weights):
            # global best across trials (even if this one gets pruned)
            prev = trial.study.user_attrs.get("best_value", float("-inf"))
            if float(value) > float(prev):
                trial.study.set_user_attr("best_value", float(value))
                trial.study.set_user_attr("best_step", int(step))
                trial.study.set_user_attr("best_trial", int(trial.number))
                trial.study.set_user_attr("best_params", weights)

                # overwrite final file only when GLOBAL best improves
                with open(f"best_params_{self.base_config["nb_jobs"]}_k{self.base_config["k"]}.json", "w") as f:
                    best_dict = {}
                    best_dict.update(weights)
                    best_dict.update({"trial": trial.number})
                    best_dict.update({"best_value": float(value)})
                    best_dict.update(self.base_config["hypers"])
                    json.dump(best_dict, f, indent=2)

    # ---------------- RESEARCH SPACE ---------------- #
    def _sample_hparams(self, trial):
        n_steps = trial.suggest_categorical("n_steps", [256, 512])
        num_minibatches = trial.suggest_categorical("num_minibatches", [1, 2])
        batch_size = n_steps // num_minibatches
        n_epochs = trial.suggest_categorical("n_epochs", [6, 8])

        return {
            "device": "cuda",
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "num_minibatches": num_minibatches,
            "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
            "ent_coef": trial.suggest_float("ent_coef", 0.01, 0.03, log=True),
            "gamma": trial.suggest_float("gamma", 0.94, 0.97),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.92, 0.95),
            "lr_initial": trial.suggest_float("lr_initial", 1e-4, 1e-3, log=True),
            "lr_final": trial.suggest_float("lr_final", 1e-6, 5e-5, log=True),
            "warmup_frac": trial.suggest_float("warmup_frac", 0, 0.08),
            "start_at": trial.suggest_float("start_at", 0, 0.30),
            #"target_kl": 0.02
        }

    def _sample_weights(self, trial):
        beta1 = trial.suggest_float("beta1", 0.0001, 1.0, log=True)
        #beta2 = trial.suggest_float("beta2", 0.01, 1.0, log=True)
        print(f"beta1: {beta1}")
        #print(f"beta2: {beta2}")

        return {
            "beta1": beta1,
            #"beta2": beta2
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASHA tuning per Reinforce")
    parser.add_argument("--config", default="UIR/config/run.yaml", help="Path al file YAML")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--grace-period", type=int, default=60_000)
    parser.add_argument("--reduction-factor", type=int, default=2)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tuner = ASHAReinforceTuner(
        base_config=cfg,
        total_steps=args.total_steps,
        eval_freq=args.eval_freq,
        n_trials=args.trials,
        grace_period=args.grace_period,
        reduction_factor=args.reduction_factor,
        n_jobs=args.n_jobs
    )

    best = tuner.tune()
    with open(f"best_params_{cfg["nb_jobs"]}_k{cfg["k"]}.json", "w") as f:
        json.dump(best, f, indent=2)
    #print("\nBest hyperparameters salvati in best_params_k2.json:\n", best)
