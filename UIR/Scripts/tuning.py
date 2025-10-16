import copy
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from collections import deque

from Dataset import Dataset
from Reinforce import Reinforce
import json
import argparse
import yaml

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
        pruner = SuccessiveHalvingPruner(
            min_resource=self.min_resource,
            reduction_factor=self.reduction_factor,
            bootstrap_count=0 #2 * self.n_jobs,  # Per stabilizzare il pruning iniziale
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed,
                                               n_startup_trials=9,
                                               multivariate=True,
                                               group=True
                                               ),
            pruner=pruner,
        )

        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        return self._sample_hparams(study.best_trial)

    # ---------------- FUNZIONE OBIETTIVO ---------------- #
    def _objective(self, trial):
        # 1) campionamento iperparametri
        hparams = self._sample_hparams(trial)

        # 2) prepara config e dataset
        cfg = copy.deepcopy(self.base_config)
        cfg["total_steps"] = self.total_steps
        cfg["eval_freq"] = self.eval_freq
        cfg["seed"] = cfg.get("seed", self.seed) + trial.number

        dataset = Dataset(cfg)

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
            beta1=0.1,
            beta2=0.9,
            params=hparams
        )

        # 4) collega Optuna al callback (solo se patcheremo EvaluateCallback)
        '''def report_fn(step, avg):
            trial.report(avg, step=step)
            return not trial.should_prune()'''
        # --- ES knobs (exploration-safe) ---
        WARMUP_EVALS        = max(3, self.min_resource)  # disable ES for first N evals
        ES_DELTA            = 0.03                      # minimal significant improvement
        CONFIRM_K           = 2                         # confirmations to accept new best
        BASE_PATIENCE       = 8                         # patience when variance is low
        HIGH_VAR_PATIENCE   = 12                        # patience when variance is high
        VAR_WINDOW          = 6                         # window for short-term std
        VAR_THRESHOLD       = 0.05                      # high-variance threshold
        COOLDOWN_AFTER_BEST = 2                         # no-stop window after a new best
        # ------------------------------------

        best = float("-inf")
        reports = 0
        min_reports = self.min_resource                # evaluations before any pruning
        min_steps   = self.min_resource * self.eval_freq
        stale = 0                                      # consecutive non-improvements
        to_prune = False

        hist_raw = deque(maxlen=VAR_WINDOW)            # keep last raw evals
        confirm_streak = 0                             # confirmations over (best+delta)
        since_best_reports = 0
        cooldown = 0

        def _update_study_best(step, value):
            # global best across trials (even if this one gets pruned)
            prev = trial.study.user_attrs.get("best_value", float("-inf"))
            if float(value) > float(prev):
                trial.study.set_user_attr("best_value", float(value))
                trial.study.set_user_attr("best_step", int(step))
                trial.study.set_user_attr("best_trial", int(trial.number))
                trial.study.set_user_attr("best_params", hparams)
                # overwrite final file only when GLOBAL best improves
                with open("best_params_.json", "w") as f:
                    json.dump(hparams, f, indent=2)

        def report_fn(step, avg):
            """Report raw metric and drive local ES in an exploration-safe way."""
            nonlocal best, reports, stale, to_prune, confirm_streak, since_best_reports, cooldown

            current = float(avg)            # raw metric: average job applicability
            hist_raw.append(current)

            # short-term variance (std) to adapt patience
            if len(hist_raw) >= 2:
                m = sum(hist_raw) / len(hist_raw)
                std = (sum((x - m) ** 2 for x in hist_raw) / len(hist_raw)) ** 0.5
            else:
                std = 0.0
            patience = HIGH_VAR_PATIENCE if std > VAR_THRESHOLD else BASE_PATIENCE

            # improvement proposal with confirmation to ignore single positive spikes
            improved = current > (best + ES_DELTA)
            if improved:
                confirm_streak += 1
                if confirm_streak >= CONFIRM_K:
                    best = current
                    confirm_streak = 0
                    stale = 0
                    since_best_reports = 0
                    cooldown = COOLDOWN_AFTER_BEST     # give space to explore around new best
                    _update_study_best(step, current)
                else:
                    # not yet accepted: treat as non-improvement for ES accounting
                    stale += 1
                    since_best_reports += 1
            else:
                if current == best:
                    confirm_streak += 1
                else:
                    confirm_streak = 0
                stale += 1
                since_best_reports += 1

            # report raw metric to Optuna/ASHA (resource = eval index)
            trial.report(current, step=reports)
            reports += 1

            # grace/warmup: block pruning before enough evaluations/steps
            if reports <= WARMUP_EVALS or step < min_reports:
                return True

            # cooldown right after a new best: avoid premature stop while exploring
            if cooldown > 0:
                cooldown -= 1
                return True

            # local ES (plateau) after warmup + cooldown
            if stale >= patience:
                to_prune = True
                return False

            # also allow ASHA to decide pruning
            return not trial.should_prune()


        if hasattr(model, "eval_callback"):
            model.eval_callback.report_fn = report_fn

        # 5) avvia training
        model.reinforce_recommendation()

        # 6) leggi la metrica finale dal callback
        avg = getattr(model.eval_callback, "last_avg_jobs", None)
        if avg is None:
            raise optuna.TrialPruned()
        
        if getattr(model.eval_callback, "was_pruned", False):
            to_prune = True

        if to_prune:
            raise optuna.TrialPruned()

        return avg

    # ---------------- SPAZIO DI RICERCA ---------------- #
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASHA tuning per Reinforce")
    parser.add_argument("--config", default="UIR/config/run.yaml", help="Path al file YAML")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--grace-period", type=int, default=60_000)
    parser.add_argument("--reduction-factor", type=int, default=2)
    parser.add_argument("--n_jobs", type=int, default=2)
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
    with open("best_params_.json", "w") as f:
        json.dump(best, f, indent=2)
    print("\nBest hyperparameters salvati in best_params_k2.json:\n", best)
