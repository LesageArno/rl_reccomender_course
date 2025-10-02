import os
import json

import numpy as np
from time import process_time

import torch
from stable_baselines3 import DQN, A2C, PPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from CourseRecEnv import CourseRecEnv, EvaluateCallback


class Reinforce:
    """Reinforcement Learning-based Course Recommendation System.
    
    This class implements a reinforcement learning approach for course recommendations
    using various RL algorithms from stable-baselines3. The system can operate in two modes:
    1. Baseline: Uses number of applicable jobs as reward
    2. Skip-expertise: Uses a utility function that considers both skill acquisition
       and job applicability
    
    The system trains an RL agent to recommend courses to learners with the goal of
    maximizing their job opportunities. The agent learns a policy that maps learner
    skill profiles to course recommendations.
    
    Attributes:
        dataset: Dataset object containing learners, jobs, and courses data
        model_name (str): Name of the RL algorithm to use ('dqn', 'a2c', or 'ppo')
        k (int): Maximum number of course recommendations per learner
        threshold (float): Minimum matching score required for job applicability
        run (int): Run identifier for experiment tracking
        total_steps (int): Total number of training steps
        eval_freq (int): Frequency of model evaluation during training
        feature (str): Feature type for reward calculation
        baseline (bool): Whether to use baseline reward (True) or utility-based reward (False)
    """
    
    def __init__(
        self, dataset, model, k, threshold, run, save_name, total_steps=1000, eval_freq=100, feature="Usefulness-of-info-as-Rwd", baseline=False, method=1, beta1=None, beta2=None
    ):  
        """Initialize the reinforcement learning recommendation system.
        
        Args:
            dataset: Dataset object containing the recommendation system data
            model (str): Name of the RL algorithm ('dqn', 'a2c', or 'ppo')
            k (int): Maximum number of course recommendations per learner
            threshold (float): Minimum matching score for job applicability
            run (int): Run identifier for experiment tracking
            total_steps (int, optional): Total training steps. Defaults to 1000.
            eval_freq (int, optional): Evaluation frequency. Defaults to 100.
            feature (str, optional): Feature type for reward. Defaults to "Usefulness-of-info-as-Rwd".
            method (int, optional): Whether to use Strict mastery or deficit based UIR function. Defaults to 1.
            baseline (bool, optional): Whether to use baseline reward. Defaults to False.
            beta1 (float, optional): Weight for job applications in reward calculation. Defaults to None.
            beta2 (float, optional): Weight for utility in reward calculation. Defaults to None.
        """
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
        # Create the training and evaluation environments
        self.train_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k, baseline = self.baseline, method=self.method, feature=self.feature, beta1=self.beta1, beta2=self.beta2)
        self.eval_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k, baseline = self.baseline, method=self.method, feature=self.feature, beta1=self.beta1, beta2=self.beta2)

        # Masking of unavailable actions
        if self.model_name == "ppo_mask":
            def mask_fn(env):  # ActionMasker function
                return env.unwrapped.get_action_mask().astype(bool)  # env.get_action_mask()

            self.train_env = ActionMasker(self.train_env, mask_fn)
            self.eval_env = ActionMasker(self.eval_env, mask_fn)

        self.get_model()
        self.all_results_filename = (
            save_name
            + "_k"
            + str(self.k)
            + "_seed"
            + str(self.dataset.config['seed'])
            + ".txt"
        )
        self.final_results_filename = (
            save_name
            + "_k"
            + str(self.k)
            + "_seed"
            + str(self.dataset.config['seed'])
            + ".json"
        )
        '''if self.baseline: #baseline model
            self.all_results_filename = (
                "all_"
                + self.model_name
                + "_skip-expertise"
                + "_nbskills_"
                + str(len(self.dataset.skills))
                + "_k_"
                + str(self.k)
                + "_run_"
                + str(run)
                + ".txt"
            )
            self.final_results_filename = (
                "final_"
                + self.model_name
                + "_skip-expertise"
                + "_nbskills_"
                + str(len(self.dataset.skills))
                + "_k_"
                + str(self.k)
                + "_run_"
                + str(self.run)
                + ".json"
            )
                
        else : ##### feature model
            self.all_results_filename = (
                "all_"
                + self.model_name
                +"_"
                + self.feature
                + "_nbskills_"
                + str(len(self.dataset.skills))
                + "_k_"
                + str(self.k)
                + "_run_"
                + str(run)
                + ".txt")
            self.final_results_filename = (
                "final_"
                + self.model_name
                + "_"
                + self.feature
                + "_nbskills_"
                + str(len(self.dataset.skills))
                + "_k_"
                + str(self.k)
                + "_run_"
                + str(self.run)
                + ".json")'''
            

        self.eval_callback = EvaluateCallback(
            self.eval_env,
            eval_freq=self.eval_freq,
            all_results_filename=self.all_results_filename,
        )

    '''def get_model(self):
        """Initialize the reinforcement learning model.
        
        Sets up the specified RL algorithm (DQN, A2C, or PPO) with default parameters.
        The model is configured to use a Multi-Layer Perceptron (MLP) policy.
        """
        # on training env
        if self.model_name == "dqn":
            self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")
        elif self.model_name == "a2c":
            self.model = A2C(env=self.train_env, verbose=0, policy="MlpPolicy", device="cpu")
        elif self.model_name == "ppo":
            self.model = PPO(env=self.train_env, verbose=0, policy="MlpPolicy")'''

    def get_model(self):
        """Initialize or load the reinforcement learning model.

        If a pretrained model path is provided in the dataset config, load the model.
        Otherwise, initialize a new model with default parameters.
        The model is configured to use a Multi-Layer Perceptron (MLP) policy.

        Supported algorithms:
        - DQN: Deep Q-Network for discrete action spaces
        - A2C: Advantage Actor-Critic
        - PPO: Proximal Policy Optimization
        """
        pretrained_path = self.dataset.config.get("pretrained_model_path", None)
        use_pretrained = self.dataset.config.get("use_pretrained", False)

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
                    custom_objects={
                        "n_steps": 2048,  # ↑ più dati per update
                        "batch_size": 1024,  # deve dividere n_envs*n_steps
                        "clip_range": 0.25,  # un filo più ampio
                        "ent_coef": 0.005,
                        "learning_rate": 2e-3,  # fine-tuning più cauto
                        "target_kl": 0.02,
                        "gae_lambda": 0.95,
                        "gamma": 0.90,
                        "seed": 42,
                        "verbose": 1
                    },
                )
                print(f"Loaded pretrained PPO model from {pretrained_path}")
            else:
                self.model = PPO(env=self.train_env,
                                 verbose=0,
                                 seed=42,
                                 policy="MlpPolicy"
                                 )

        elif self.model_name == "ppo_mask":
            if use_pretrained:
                self.model = MaskablePPO.load(
                    pretrained_path,
                    env=self.train_env,
                    device="auto",
                    custom_objects={
                        "n_steps": 2048,  # ↑ più dati per update
                        "batch_size": 1024,  # deve dividere n_envs*n_steps
                        "clip_range": 0.25,  # un filo più ampio
                        "ent_coef": 0.005,
                        "learning_rate": 2e-3,  # fine-tuning più cauto
                        "target_kl": 0.02,
                        "gae_lambda": 0.95,
                        "gamma": 0.90,
                        "seed": 42,
                        "verbose": 1
                    },
                )
            else:
                self.model = MaskablePPO(
                    "MlpPolicy",
                    env=self.train_env,
                    device="auto",
                    seed=42,
                    gamma=0.99,
                    n_steps=2048,
                    batch_size=1024,
                    ent_coef=0.02,
                    clip_range=0.2,
                    verbose=0,
                )

                '''self.model = MaskablePPO(
                    "MlpPolicy",
                    env=self.train_env,
                    device="auto",
                    seed=42
                )'''

                #self.model = MaskablePPO(env=self.train_env, verbose=0, policy="MlpPolicy")
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def update_learner_profile(self, learner, course):
        """Updates the learner's profile with the skills and levels of the course.

        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            
        Returns:
            np.ndarray: Updated learner's skill vector
        """
        learner = np.maximum(learner, course[1])
        return learner

    def reinforce_recommendation(self):
        """Train and evaluate the RL model for course recommendations.
        
        This method:
        1. Calculates initial metrics (attractiveness and applicable jobs)
        2. Trains the RL model using the training environment
        3. Evaluates the model on all learners
        4. Generates course recommendations for each learner
        5. Updates learner profiles based on recommendations
        6. Calculates final metrics and saves results_k2
        
        The results_k2 are saved in two files:
        - A text file with intermediate evaluation results_k2
        - A JSON file with final metrics and recommendations
        """
        results = dict()

        avg_l_attrac_debut = self.dataset.get_avg_learner_attractiveness() #debut
        print(f"The average attractiveness of the learners is {avg_l_attrac_debut:.2f}")
        

        results["original_attractiveness"] = avg_l_attrac_debut

        avg_app_j_debut = self.dataset.get_avg_applicable_jobs(self.threshold) #debut
        print(f"The average nb of applicable jobs per learner is {avg_app_j_debut:.2f}")
        
        results["original_applicable_jobs"] = avg_app_j_debut

        # Train the model using train env
        self.model.learn(total_timesteps=self.total_steps, callback=self.eval_callback, log_interval=10)# find the policy

        # Save model after training
        save_dir = self.dataset.config.get("save_dir", "UIR")
        model_dir = os.path.join(save_dir, "models_weights")
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, self.final_results_filename))

        # Evaluate the model using eval env
        time_start = process_time()
        recommendations = dict()
        for i, learner in enumerate(self.dataset.learners):#run by row
            self.eval_env.reset(options={"learner":learner}) #initialize _agent_skills = learner if not NONE
            done = False
            index = self.dataset.learners_index[i]
            recommendation_sequence = []
            while not done:
                obs = self.eval_env.unwrapped.get_obs() #return _agent_skills which is current state
                if isinstance(self.model, MaskablePPO):
                    mask = self.eval_env.unwrapped.get_action_mask()
                    action, _state = self.model.predict(obs, action_masks=mask, deterministic=True)
                else:
                    action, _state = self.model.predict(obs, deterministic=True)  # Predict action using current policy
                # action is Recommended course index [0,99]action_space
                obs, reward, done, _, info = self.eval_env.step(action)
                if reward != -1:
                    recommendation_sequence.append(action.item())
            for course in recommendation_sequence:
                self.dataset.learners[i] = self.update_learner_profile(
                    learner, self.dataset.courses[course]
                )

            recommendations[index] = [
                self.dataset.courses_index[course_id]
                for course_id in recommendation_sequence
            ]

        time_end = process_time()
        avg_recommendation_time = (time_end - time_start) / len(self.dataset.learners)

        print(f"Average Recommendation Time: {avg_recommendation_time:.4f} seconds")
        results["avg_recommendation_time"] = avg_recommendation_time

        avg_l_attrac_fin = self.dataset.get_avg_learner_attractiveness() #fin
        print(f"The new average attractiveness of the learners is {avg_l_attrac_fin:.2f}")
    

        results["new_attractiveness"] = avg_l_attrac_fin

        avg_app_j_fin = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The new average nb of applicable jobs per learner is {avg_app_j_fin:.2f}")
        


        results["new_applicable_jobs"] = avg_app_j_fin

        results["recommendations"] = recommendations

        json.dump(
            results,
            open(
                os.path.join(
                    f"{self.dataset.config['results_path']}_k{self.k}_seed{self.dataset.config['seed']}",
                    self.final_results_filename,
                ),
                "w",
            ),
            indent=4,
        )
