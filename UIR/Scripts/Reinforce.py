import os
import json
import math
from time import process_time

import numpy as np
import torch
from stable_baselines3 import DQN, A2C, PPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from CourseRecEnv import CourseRecEnv, EvaluateCallback


class Reinforce:
    """Reinforcement Learning-based Course Recommendation System.

    This class implements a reinforcement learning approach for course recommendations
    using various RL algorithms from stable-baselines3. The system can operate in two modes:
      1) Baseline: uses number of applicable jobs as reward
      2) Skip-expertise (utility-based): considers both skill acquisition and job applicability

    The goal is to train an RL agent that recommends courses to maximize learners' job opportunities.

    Attributes:
        dataset: Dataset object containing learners, jobs, and courses
        model_name (str): RL algorithm ('dqn', 'a2c', 'ppo', or 'ppo_mask')
        k (int): Max number of course recommendations per learner
        threshold (float): Min matching score required for job applicability
        run (int): Run identifier for experiment tracking
        save_name (str): Base name used when saving results
        total_steps (int): Total number of training steps
        eval_freq (int): Frequency of evaluation during training
        feature (str): Feature type for reward calculation
        baseline (bool): Whether to use baseline reward
        method (int): Mastery method flag (strict vs deficit-based)
        beta1, beta2 (float|None): Weights for weighted reward (optional)
        params (dict|None): Algorithm hyperparameters (optional)
    """

    def __init__(
        self,
        dataset,
        model,
        k,
        threshold,
        run,
        save_name,
        total_steps=1000,
        eval_freq=100,
        feature="Usefulness-of-info-as-Rwd",
        baseline=False,
        method=1,
        beta1=None,
        beta2=None,
        params=None,
    ):
        """Initialize the RL recommendation system (no changes in behavior)."""
        self.baseline = baseline
        self.method = max(0, min(1, method))
        self.dataset = dataset
        self.model_name = model
        self.k = k
        self.threshold = threshold
        self.run = run
        self.save_name = save_name
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        self.feature = feature
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = params

        # Create training and evaluation environments
        self.train_env = CourseRecEnv(
            dataset,
            threshold=self.threshold,
            k=self.k,
            baseline=self.baseline,
            method=self.method,
            feature=self.feature,
            beta1=self.beta1,
            beta2=self.beta2,
        )
        self.eval_env = CourseRecEnv(
            dataset,
            threshold=self.threshold,
            k=self.k,
            baseline=self.baseline,
            method=self.method,
            feature=self.feature,
            beta1=self.beta1,
            beta2=self.beta2,
        )

        # Mask unavailable actions when using maskable PPO
        if self.model_name == "ppo_mask":

            def mask_fn(env):
                # Return boolean action mask from the unwrapped environment
                return env.unwrapped.get_action_mask().astype(bool)

            self.train_env = ActionMasker(self.train_env, mask_fn)
            self.eval_env = ActionMasker(self.eval_env, mask_fn)

        self.get_model()

        self.all_results_filename = (
            save_name
            + "_k"
            + str(self.k)
            + "_seed"
            + str(self.dataset.config["seed"])
            + ".txt"
        )
        self.final_results_filename = (
            save_name
            + "_k"
            + str(self.k)
            + "_seed"
            + str(self.dataset.config["seed"])
            + ".json"
        )

        self.eval_callback = EvaluateCallback(
            self.eval_env,
            eval_freq=self.eval_freq,
            all_results_filename=self.all_results_filename,
        )

    def get_model(self):
        """Initialize or load the RL model with an MLP policy (no semantic changes).

        Supported algorithms:
          - DQN: Deep Q-Network (discrete actions)
          - A2C: Advantage Actor-Critic
          - PPO / MaskablePPO: Proximal Policy Optimization (with/without action masking)
        """

        def delayed_cosine_schedule(
            initial: float, final: float, start_at: float = 0.65, warmup_frac: float = 0.05
        ):
            """Piecewise schedule: warmup -> flat -> cosine decay to final."""
            initial = float(initial)
            final = float(final)

            def sched(progress_remaining: float):
                elapsed = 1.0 - progress_remaining
                if elapsed <= warmup_frac:
                    return initial * (elapsed / max(1e-12, warmup_frac))
                if elapsed <= start_at:
                    return initial
                # Map to [0, pi] for cosine
                frac = (elapsed - start_at) / (1.0 - start_at)
                cos_part = 0.5 * (1 + math.cos(math.pi * frac))
                return final + (initial - final) * cos_part

            return sched

        pretrained_path = self.dataset.config.get("pretrained_model_path", None)
        use_pretrained = self.dataset.config.get("use_pretrained", False)

        if self.params is None:
            self.params = {
                "gamma": 0.95,
                "gae_lambda": 0.93,
                "n_steps": 256 if self.k < 3 else 512,
                "batch_size": 128 if self.k < 3 else 256,
                "n_epochs": 10,
                "lr_initial": 1e-3,
                "lr_final": 5e-5,
                "warmup_frac": 0.05,
                "start_at": 0.05,
                "ent_coef": 0.025,
                "clip_range": 0.2,
                "device": "auto",
            }

        if self.model_name == "dqn":
            if use_pretrained:
                self.model = DQN.load(pretrained_path, env=self.train_env)
                print(f"Loaded pretrained DQN model from {pretrained_path}")
            else:
                self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")

        elif self.model_name == "a2c":
            if use_pretrained:
                self.model = A2C.load(pretrained_path, env=self.train_env, device="cpu")
                print(f"Loaded pretrained A2C model from {pretrained_path}")
            else:
                self.model = A2C(env=self.train_env, verbose=0, policy="MlpPolicy", device="cpu")

        elif self.model_name == "ppo":
            if use_pretrained:
                self.model = PPO.load(
                    pretrained_path,
                    env=self.train_env,
                    device="auto",
                    custom_objects=self.params,
                )
                print(f"Loaded pretrained PPO model from {pretrained_path}")
            else:
                if self.baseline:
                    self.model = PPO(
                        "MlpPolicy",
                        env=self.train_env,
                        seed=self.dataset.config["seed"],
                        verbose=0,
                    )
                else:
                    self.model = PPO(
                        "MlpPolicy",
                        env=self.train_env,
                        device="auto",
                        seed=self.dataset.config["seed"],
                        gamma=self.params["gamma"],
                        gae_lambda=self.params["gae_lambda"],
                        n_steps=self.params["n_steps"],
                        batch_size=self.params["batch_size"],
                        n_epochs=self.params["n_epochs"],
                        learning_rate=delayed_cosine_schedule(
                            self.params["lr_initial"],
                            self.params["lr_final"],
                            start_at=self.params["start_at"],
                            warmup_frac=self.params["warmup_frac"],
                        ),
                        ent_coef=self.params["ent_coef"],
                        clip_range=self.params["clip_range"],
                        target_kl=self.params.get("target_kl", None),
                        verbose=0,
                    )

        elif self.model_name == "ppo_mask":
            if use_pretrained:
                self.model = MaskablePPO.load(
                    pretrained_path,
                    env=self.train_env,
                    device="auto",
                    custom_objects=self.params,
                )
            else:
                print("CUDA Available: ", torch.cuda.is_available())
                self.model = MaskablePPO(
                    "MlpPolicy",
                    env=self.train_env,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    seed=self.dataset.config["seed"],
                    gamma=self.params["gamma"],
                    gae_lambda=self.params["gae_lambda"],
                    n_steps=self.params["n_steps"],
                    batch_size=self.params["batch_size"],
                    n_epochs=self.params["n_epochs"],
                    learning_rate=delayed_cosine_schedule(
                        self.params["lr_initial"],
                        self.params["lr_final"],
                        start_at=self.params["start_at"],
                        warmup_frac=self.params["warmup_frac"],
                    ),
                    ent_coef=self.params["ent_coef"],
                    clip_range=self.params["clip_range"],
                    target_kl=self.params.get("target_kl", None),
                    verbose=0,
                )
                print(self.save_name)
                print(self.model.policy.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def update_learner_profile(self, learner, course):
        """Update learner's skill vector with the course-provided skills/levels.

        Args:
            learner (np.ndarray): Current learner skill vector.
            course (np.ndarray): Course skills array [required, provided].

        Returns:
            np.ndarray: Updated learner skill vector.
        """
        learner = np.maximum(learner, course[1])
        return learner

    def reinforce_recommendation(self):
        """Train and evaluate the RL model; compute and save results.

        Steps:
          1) Compute initial metrics (attractiveness and applicable jobs)
          2) Train the RL agent on the training environment
          3) Evaluate the trained policy over all learners
          4) Produce a recommendation sequence per learner
          5) Update learner profiles based on recommended courses
          6) Compute final metrics and save results

        Outputs:
          - TXT file with intermediate evaluation results
          - JSON file with final metrics and recommendations
        """
        results = dict()

        # Initial metrics (start)
        avg_l_attrac_debut = self.dataset.get_avg_learner_attractiveness()
        print(f"The average attractiveness of the learners is {avg_l_attrac_debut:.2f}")
        results["original_attractiveness"] = avg_l_attrac_debut

        avg_app_j_debut = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The average nb of applicable jobs per learner is {avg_app_j_debut:.2f}")
        results["original_applicable_jobs"] = avg_app_j_debut

        # Train the model (find the policy)
        self.model.learn(total_timesteps=self.total_steps, callback=self.eval_callback, log_interval=10)

        # Optionally save the trained model
        save_dir = self.dataset.config.get("save_dir", "UIR")
        model_dir = os.path.join(save_dir, "models_weights")
        os.makedirs(model_dir, exist_ok=True)
        if self.dataset.config.get("save_model", False):
            self.model.save(os.path.join(model_dir, self.final_results_filename))

        # Evaluate the model and produce recommendations
        time_start = process_time()
        recommendations = dict()
        for i, learner in enumerate(self.dataset.learners):
            # Initialize evaluation environment with the selected learner
            self.eval_env.reset(options={"learner": learner})
            done = False
            index = self.dataset.learners_index[i]
            recommendation_sequence = []

            while not done:
                # Observation is the current learner skill state
                obs = self.eval_env.unwrapped.get_obs()
                if isinstance(self.model, MaskablePPO):
                    mask = self.eval_env.unwrapped.get_action_mask()
                    action, _state = self.model.predict(obs, action_masks=mask, deterministic=True)
                else:
                    action, _state = self.model.predict(obs, deterministic=True)

                # Step the environment with the chosen action
                obs, reward, done, _, info = self.eval_env.step(action)

                # Only record valid recommendations (reward != -1)
                if reward != -1:
                    recommendation_sequence.append(action.item())

            # Update learner profile with the recommended courses
            for course in recommendation_sequence:
                self.dataset.learners[i] = self.update_learner_profile(
                    learner, self.dataset.courses[course]
                )

            # Store course identifiers for this learner
            recommendations[index] = [
                self.dataset.courses_index[course_id] for course_id in recommendation_sequence
            ]

        time_end = process_time()
        avg_recommendation_time = (time_end - time_start) / len(self.dataset.learners)
        print(f"Average Recommendation Time: {avg_recommendation_time:.4f} seconds")
        results["avg_recommendation_time"] = avg_recommendation_time

        # Final metrics (end)
        avg_l_attrac_fin = self.dataset.get_avg_learner_attractiveness()
        print(f"The new average attractiveness of the learners is {avg_l_attrac_fin:.2f}")
        results["new_attractiveness"] = avg_l_attrac_fin

        avg_app_j_fin = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The new average nb of applicable jobs per learner is {avg_app_j_fin:.2f}")
        results["new_applicable_jobs"] = avg_app_j_fin

        results["recommendations"] = recommendations

        # Persist final results to JSON (path logic unchanged)
        json.dump(
            results,
            open(
                os.path.join(
                    f"{self.dataset.config['results_path']}{self.k}/seed{self.dataset.config['seed']}",
                    self.final_results_filename,
                ),
                "w",
            ),
            indent=4,
        )
