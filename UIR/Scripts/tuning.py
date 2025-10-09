import copy
import optuna
from optuna.pruners import SuccessiveHalvingPruner

from Dataset import Dataset
from Reinforce import Reinforce
import json
import argparse
import yaml

class ASHAReinforceTuner:
    """
    Tuning iperparametri per Reinforce con Optuna + ASHA (Successive Halving).
    """

    def __init__(self, base_config, total_steps=150_000, eval_freq=5_000,
                 n_trials=30, grace_period=10_000, reduction_factor=3):
        self.base_config = copy.deepcopy(base_config)
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.n_trials = n_trials
        self.seed = self.base_config.get('seed', 42)
        self.min_resource = grace_period // eval_freq
        self.reduction_factor = reduction_factor

    # ---------------- TUNING PRINCIPALE ---------------- #
    def tune(self):
        pruner = SuccessiveHalvingPruner(
            min_resource=self.min_resource,
            reduction_factor=self.reduction_factor
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=pruner,
        )

        study.optimize(self._objective, n_trials=self.n_trials)
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
        def report_fn(step, avg):
            trial.report(avg, step=step)
            return not trial.should_prune()

        if hasattr(model, "eval_callback"):
            model.eval_callback.report_fn = report_fn

        # 5) avvia training
        model.reinforce_recommendation()

        # 6) leggi la metrica finale dal callback
        avg = getattr(model.eval_callback, "last_avg_jobs", None)
        if avg is None:
            raise optuna.TrialPruned()

        return avg

    # ---------------- SPAZIO DI RICERCA ---------------- #
    def _sample_hparams(self, trial):
        n_steps = trial.suggest_categorical("n_steps", [512, 1024])
        num_minibatches = trial.suggest_categorical("num_minibatches", [1, 2])
        batch_size = n_steps // num_minibatches
        n_epochs = trial.suggest_categorical("n_epochs", [4, 6] if num_minibatches == 1 else [2, 4])

        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "num_minibatches": num_minibatches,
            "clip_range": trial.suggest_float("clip_range", 0.10, 0.25),
            "ent_coef": trial.suggest_float("ent_coef", 5e-4, 3e-2, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.97),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.93, 0.96),
            "lr_initial": trial.suggest_float("lr_initial", 1e-4, 5e-4, log=True),
            "lr_final": 5e-5,
            "warmup_frac": 0.05,
            "start_at": 0.05,
            "target_kl": 0.015
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASHA tuning per Reinforce")
    parser.add_argument("--config", default="UIR/config/run.yaml", help="Path al file YAML")
    parser.add_argument("--total-steps", type=int, default=150_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--grace-period", type=int, default=10_000)
    parser.add_argument("--reduction-factor", type=int, default=3)
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
    )

    best = tuner.tune()
    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("\nBest hyperparameters salvati in best_params.json:\n", best)
