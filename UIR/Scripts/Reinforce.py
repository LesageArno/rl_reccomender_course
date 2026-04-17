import os
import math
import torch

from stable_baselines3 import DQN, A2C, PPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from .CourseRecEnv import CourseRecEnv, EvaluateCallback


class Reinforce:
    """Reinforcement Learning-based Course Recommendation System.

    This class implements a reinforcement learning approach for course recommendations
    using various RL algorithms from stable-baselines3. The system can operate in two modes:
      1) Employability (baseline): uses number of applicable jobs as reward
      2) UIR (utility-based): consider Usefulness of information as reward 

    The goal is to train an RL agent that recommends courses to maximize learners' job opportunities.

    Attributes:
        dataset: Dataset object containing learners, jobs, and courses
        config: Configuration dictionary for the recommendation system
        model_name (str): RL algorithm ('dqn', 'a2c', 'ppo', or 'ppo_mask')
        k (int): Max number of course recommendations per learner
        threshold (float): Min matching score required for job applicability
        save_name (str): Base name used when saving results
        total_steps (int): Total number of training steps
        eval_freq (int): Frequency of evaluation during training
        feature (str): Feature type for reward calculation
        method (int): Mastery method flag (threshold-based vs gap-based)
        params (dict|None): Algorithm hyperparameters (optional)
    """

    def __init__(
        self,
        dataset,
        config,
        k = None,
        use_pretrained=False,
        pretrained_path=None
    ):
        """Initialize the RL recommendation system"""
        self.dataset = dataset
        self.config = config
        self.method = config["method"]
        self.model_name = config["model"]
        if k is not None:
            self.k = k
        else:
            self.k = config["k"]
        self.threshold = config["threshold"]
        self.save_name = config["name_exp"]
        self.total_steps = config["total_steps"]
        self.eval_freq = config["eval_freq"]
        self.feature = config["feature"]
        self.params = config.get("hypers", None)
        self.use_standard = config.get('use_standard', False)
        self.use_pretrained = use_pretrained or self.config.get("use_pretrained", False)
        self.pretrained_path = pretrained_path or self.config.get("pretrained_model_path")

        if self.feature in {"UIR", "EUIR"} and self.method not in (0, 1):
            raise ValueError("method must be 0 (threshold) or 1 (gap) when feature is UIR/EUIR")



        # Create training and evaluation environments
        self.train_env = CourseRecEnv(
            dataset,
            config=self.config,
            k=self.k,
        )
        self.eval_env = CourseRecEnv(
            dataset,
            config=self.config,
            k=self.k,
        )

        # Mask unavailable actions when using maskable PPO
        if self.model_name == "ppo_mask":

            def mask_fn(env):
                # Return boolean action mask from the unwrapped environment
                return env.get_wrapper_attr("get_action_mask")()

            self.train_env = ActionMasker(self.train_env, mask_fn)
            self.eval_env = ActionMasker(self.eval_env, mask_fn)

        self.get_model()

        self.all_results_filename = (
            self.save_name
            + "_k"
            + str(self.k)
            + "_seed"
            + str(self.config["seed"])
            + ".txt"
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

        def delayed_cosine_schedule(initial: float, final: float, start_at: float = 0.65, warmup_frac: float = 0.05):
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
            if self.use_pretrained:
                self.model = DQN.load(f"{self.pretrained_path}_k{self.k}", env=self.train_env)
                print(f"Loaded pretrained DQN model from {self.pretrained_path}_k{self.k}")
            else:
                self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")

        elif self.model_name == "a2c":
            if self.use_pretrained:
                self.model = A2C.load(f"{self.pretrained_path}_k{self.k}", env=self.train_env, device=self.params["device"])
                print(f"Loaded pretrained A2C model from {self.pretrained_path}_k{self.k}")
            else:
                self.model = A2C(env=self.train_env, verbose=0, policy="MlpPolicy", device=self.params["device"])

        elif self.model_name == "ppo":
            if self.use_pretrained:
                self.model = PPO.load(
                    f"{self.pretrained_path}_k{self.k}",
                    env=self.train_env,
                    device=self.params["device"],
                )
                print(f"Loaded pretrained PPO model from {self.pretrained_path}_k{self.k}")
            else:
                if self.use_standard:
                    self.model = PPO(
                        "MlpPolicy",
                        env=self.train_env,
                        seed=self.config["seed"],
                        verbose=0,
                    )
                else:
                    self.model = PPO(
                        "MlpPolicy",
                        env=self.train_env,
                        device=self.params["device"],
                        max_grad_norm=self.params["max_grad_norm"],
                        seed=self.config["seed"],
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
            if self.use_pretrained:
                self.model = MaskablePPO.load(
                    f"{self.pretrained_path}_k{self.k}",
                    env=self.train_env,
                    device=self.params["device"],
                )
                print(f"Loaded pretrained MaskablePPO model from {self.pretrained_path}_k{self.k}")
            else:
                print("CUDA Available: ", torch.cuda.is_available())
                if self.use_standard:
                    self.model = MaskablePPO(
                        'MlpPolicy',
                        env=self.train_env,
                        device=self.params["device"],
                        seed=self.config["seed"],
                        verbose=0
                    )
                else:
                    self.model = MaskablePPO(
                        "MlpPolicy",
                        env=self.train_env,
                        device=self.params["device"], #"cuda" if torch.cuda.is_available() else "cpu",
                        max_grad_norm=self.params["max_grad_norm"],
                        seed=self.config["seed"],
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

    def reinforce_recommendation(self):
        """
        Train the reinforcement learning model.

        The training process:
        - uses the configured RL algorithm (PPO, MaskablePPO, etc.)
        - logs evaluation metrics via EvaluateCallback
        - optionally saves the trained model

        All performance metrics and intermediate evaluations are handled
        by the EvaluateCallback during training.
        """
        # Train the model (find the policy)
        self.model.learn(total_timesteps=self.total_steps, 
                         callback=self.eval_callback, 
                         log_interval=10)

        # Optionally save the trained model
        if self.config.get("save_model", False):
            save_dir = self.config.get("save_dir")

            if not save_dir:
                raise ValueError("save_model=true requires 'save_dir' in config.")
            
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{self.save_name}_k{self.k}")
            self.model.save(save_path)
            
            print(f"[INFO] Model saved to: {save_path}")

    def recommend(self, learner_vec, want, avoid, forbidden=None):
        env = self.eval_env.unwrapped

        print(f"want = {want}")
        print(f"avoid = {avoid}")

        #env.set_extra_invalid_actions(forbidden)
        #obs, _ = env.reset(options={"learner": learner_vec})
        obs, _ = env.reset(options={"learner": learner_vec, "want": want, "avoid": avoid})

        seq = []
        seq_readable = []
        done = False
        nb_jobs = 0

        while not done:
            mask = env.get_action_mask()
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if reward != -1:
                seq.append(int(action))
                course_id = self.dataset.courses_index[int(action)]
                seq_readable.append(course_id)
                nb_jobs = info["nb_applicable_jobs"]

        return {
            "seq_ids": seq,
            "seq_course_codes": seq_readable,
            "nb_applicable_jobs": nb_jobs,
            "jobs_goal": env.jobs_goal
        }
