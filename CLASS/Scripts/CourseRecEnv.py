import os
import random
from pathlib import Path
import csv, uuid, time

from time import process_time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

import matchings
from clustering import CourseClusterer


class CourseRecEnv(gym.Env):
    """Course Recommendation Environment for Reinforcement Learning.
    
    This class implements a Gymnasium environment for course recommendations using
    reinforcement learning with mastery levels and optional clustering-based reward adjustment.
    
    The environment uses the number of applicable jobs as the reward signal to train
    the RL agent. The reward can be optionally adjusted based on course clustering
    to encourage more stable learning.
    
    Observation Space:
        - Vector of length nb_skills representing learner's current skill levels
        - Each element is an integer in [0, 3] where:
            * 0: No skill
            * 1: Basic mastery
            * 2: Intermediate mastery
            * 3: Advanced mastery
        - Shape: (nb_skills,)
    
    Action Space:
        - Discrete space of size nb_courses
        - Each action represents recommending a specific course
        - Range: [0, nb_courses-1]
    
    Attributes:
        dataset: Dataset object containing learners, jobs, and courses data
        nb_skills (int): Number of unique skills in the system
        mastery_levels (list): List of possible mastery levels [1,2,3]
        max_level (int): Maximum mastery level (3)
        nb_courses (int): Number of available courses
        min_skills (int): Minimum number of skills a learner can have
        max_skills (int): Maximum number of skills a learner can have
        threshold (float): Minimum matching score required for job applicability
        k (int): Maximum number of course recommendations per learner
        use_clustering (bool): Whether to use clustering for reward adjustment
    """
    
    def __init__(self, dataset, threshold=0.5, k=1, is_training=False):
        """Initialize the course recommendation environment.
        
        Args:
            dataset: Dataset object containing learners, jobs, and courses
            threshold (float): Minimum matching score for job applicability
            k (int): Maximum number of course recommendations per learner
            is_training (bool): Whether this is a training environment
        """
        print(f"\nInitializing CourseRecEnv:")
        print(f"use_clustering: {hasattr(dataset, 'config') and dataset.config.get('use_clustering', False)}")
        print(f"is_training: {is_training}")
    

        self.dataset = dataset
        self.cfg = dataset.config
        self.threshold = threshold
        self.k = k
        self.is_training = is_training
        
        # Initialize basic attributes
        self.band = self.dataset.config.get('band', 0.1)
        self.w_cross = self.dataset.config.get('w_cross', 1)
        self.w_final_app = self.dataset.config.get('w_final_app', 1)
        self.w_setup = self.dataset.config.get('w_setup', 1)
        self.w_margin = self.dataset.config.get('w_margin', 1)
        self.w_drop = self.dataset.config.get('w_drop', 1)
        self.w_zero_cross = self.dataset.config.get('zero_cross_weight', 1)
        self.w_zero_app = self.dataset.config.get('zero_app_weight', 1)
        self.obs_shape = self.dataset.config.get('obs_shape', 51)
        self.nb_skills = len(dataset.skills)  # 46 skills
        self.mastery_levels = [1, 2, 3]
        self.max_level = 3
        self.nb_courses = len(dataset.courses)  # 100 courses
        self.min_skills = min(np.count_nonzero(self.dataset.learners, axis=1))  # 1
        self.max_skills = max(np.count_nonzero(self.dataset.learners, axis=1))  # 15
        
        # Initialize observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_level, shape=(self.obs_shape,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(self.nb_courses)
        
        # Initialize clustering only in training environment
        self.use_clustering = False
        self.clusterer = None
        self.prev_reward = None
        
        if self.is_training and hasattr(dataset, 'config') and dataset.config.get("use_clustering", False):
            self.use_clustering = True
            self.clusterer = CourseClusterer(
                n_clusters=dataset.config.get("n_clusters", 5),
                random_state=dataset.config.get("seed", 42),
                auto_clusters=dataset.config.get("auto_clusters", False),
                max_clusters=dataset.config.get("max_clusters", 10),
                config=dataset.config.get("clustering", {}),
                dataset=self.dataset
            )
            # Fit clusters in training environment
            if self.clusterer.course_clusters is None:
                self.clusterer.fit_course_clusters(dataset.courses)
        
        # Initialize environment state
        self.reset()

        # --- Logging setup ---
        self.enable_cluster_log = bool(self.cfg.get("enable_cluster_log", True))
        self.run_id = self.cfg.get("run_name", f"run_{int(time.time())}")
        self.cluster_log_path = Path(self.cfg.get("out_dir", "results")) / self.run_id / "cluster_log.csv"
        self.cluster_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode_id = None
        if self.enable_cluster_log and not self.cluster_log_path.exists():
            with open(self.cluster_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode_id", "step", "course_idx", "cluster_id", "base_reward", "adjusted_reward"])

    def get_obs(self):
        """Get the current observation of the environment.
        
        Returns:
            np.ndarray: Current learner's skill vector representing the state
        """
        skills = (self._agent_skills.astype(np.float32) / 3.0)
        step_left = np.array([(self.k - self.nb_recommendations) / self.k])

        # near_frac: quota di job in [T-band, T)
        m = matchings.matches_array(self._agent_skills, self.dataset.jobs)
        near_mask = (m >= self.threshold - self.band) & (m < self.threshold)
        near_frac = np.array([near_mask.mean()])

        applicable_now = np.array([np.mean(m >= self.threshold)])

        gaps = np.clip(self.threshold - m, 0.0, 1.0)
        min_gap = np.array([gaps.min() if gaps.size else 1.0], dtype=np.float32)

        near_gaps = gaps[near_mask]
        mean_gap_near = np.array([near_gaps.mean() if near_gaps.size else 1.0], dtype=np.float32)

        obs = np.concatenate([skills, step_left, near_frac, applicable_now, min_gap, mean_gap_near], axis=0)

        return obs

    def get_info(self):
        """Get additional information about the current state.
        
        Returns:
            dict: Dictionary containing the number of applicable jobs for the current state
        """
        return {
            "nb_applicable_jobs": self.dataset.get_nb_applicable_jobs(
                self._agent_skills, threshold=self.threshold
            )
        }

    def get_random_learner(self):
        """Generate a random learner profile for environment initialization.
        
        Creates a learner with:
        - Random number of skills between min_skills and max_skills
        - Random mastery levels for each skill
        
        Returns:
            np.array: the initial observation of the environment, that is the learner's initial skills
        """
        # Randomly choose the number of skills the agent has randomly
        n_skills = random.randint(self.min_skills, self.max_skills)

        # Initialize the skills array with zeros
        initial_skills = np.zeros(self.nb_skills, dtype=np.int32)

        # Choose unique skill indices without replacement
        skill_indices = np.random.choice(self.nb_skills, size=n_skills, replace=False)

        # Assign random mastery levels to these skills, levels can repeat
        initial_skills[skill_indices] = np.random.choice(
            self.mastery_levels, size=n_skills, replace=True
        )
        return initial_skills

    def reset(self, seed=None, options=None, learner=None):
        """Method required by the gym environment. It resets the environment to its initial state.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            learner (np.ndarray, optional): Initial learner profile. If None, generates random profile. Defaults to None.
            
        Returns:
            tuple: (observation, info) where:
                - observation: Initial learner's skill vector
                - info: Dictionary containing initial state information
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._episode_id = str(uuid.uuid4())  # nuovo episodio
        self._step_in_ep = 0
        if options is not None:
            self._agent_skills = options["learner"]

        if learner is not None:
            self._agent_skills = learner
        else:
            self._agent_skills = self.get_random_learner()
        self.nb_recommendations = 0

        m0 = matchings.matches_array(self._agent_skills, self.dataset.jobs)
        band = 0.10  # lo stesso usato nella reward
        self._opp0 = int(((m0 >= self.threshold - band) & (m0 < self.threshold)).sum())
        self._crossings_total = 0

        # Reset clustering-related attributes
        self.prev_reward = None
        if self.use_clustering and self.clusterer is not None:
            self.clusterer.prev_cluster = None
            
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def get_action_mask(self) -> np.ndarray:
        """
        True = valid actions, False = invalid actions.
        Invalid:
          - required_matching < threshold
          - provided_matching >= 1.0
        """
        learner = self._agent_skills
        valid = np.ones(self.dataset.courses.shape[0], dtype=bool)

        for i, course in enumerate(self.dataset.courses):
            req = matchings.learner_course_required_matching(learner, course)
            prov = matchings.learner_course_provided_matching(learner, course)
            if req < self.threshold or prov >= 1.0:
                valid[i] = False

        if not valid.any():  # fallback difensivo
            valid[0] = True
        return valid

    def step(self, action):
        """Execute one step in the environment.
        
        This method:
        1. Recommends a course based on the action
        2. Updates the learner's skills if the course is valid
        3. Calculates the reward based on number of applicable jobs
        4. Adjusts reward using clustering if enabled (only in training)
        5. Checks if the episode should terminate
        
        Args:
            action (int): Index of the course to recommend
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info) where:
                - observation: Updated learner's skill vector
                - reward: Number of applicable jobs
                - terminated: Whether the episode is done
                - truncated: Whether the episode was truncated
                - info: Additional information about the step
        """
        course = self.dataset.courses[action]
        learner = self._agent_skills

        prev_skills = self._agent_skills.copy()

        # Skip if learner already has all skills provided by the course
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        required_matching = matchings.learner_course_required_matching(learner, course)
        if required_matching < self.threshold or provided_matching >= 1.0:
            observation = self.get_obs()
            reward = -1
            terminated = True
            info = self.get_info()
            print(f"Ho scelto l'azione {action} ma non va. MALEDIZIONE AL DEMONIO")
            return observation, reward, terminated, False, info

        # print(f"This is value matching BEFORE a course was selected {matchings.compute_coverage_percentage(self._agent_skills, self.dataset.jobs)}")
        
        # Update learner's skills
        self._agent_skills = np.maximum(self._agent_skills, course[1])
        observation = self.get_obs()
        info = self.get_info()

        post_skills = self._agent_skills.copy()

        is_last = self.k == self.nb_recommendations
        reward, logs = matchings.action_reward_per_course(
            prev_learner=prev_skills,
            next_learner=post_skills,
            jobs=self.dataset.jobs,
            threshold=self.threshold,
            band=self.band,
            w_cross=self.w_cross,
            w_final_app=self.w_final_app,
            w_setup=self.w_setup,  # segnala il potenziale per lo step successivo
            w_margin=self.w_margin,
            w_drop=self.w_drop,
            is_last=is_last,
            ep_crossings_so_far=self._crossings_total,
            w_zero_cross=self.w_zero_cross,
            w_zero_app=self.w_zero_app,
            opp0=self._opp0,
            opp_ref=5
        )
        # print("Reward: ", reward)

        '''# Set reward as number of applicable jobs
        reward = info["nb_applicable_jobs"]'''
        original_reward = reward
        # print(f"This is the reward {reward}")
        # print(f"This is value matching after a course was selected {matchings.compute_coverage_percentage(self._agent_skills, self.dataset.jobs)}")
        # Adjust reward using clustering only in training environment
        if self.use_clustering and self.clusterer is not None and self.is_training:
            reward = self.clusterer.adjust_reward(
                course_idx=action,
                original_reward=reward,
                prev_reward=self.prev_reward,
                actual_skills=self._agent_skills
            )
            self.prev_reward = reward

        '''# --- Cluster logging ---
        if self.enable_cluster_log and hasattr(self, "clusterer") and self.clusterer is not None:
            course_idx = int(action)
            cluster_id = int(self.clusterer.course_clusters[course_idx])
            with open(self.cluster_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self._episode_id,
                    self._step_in_ep,
                    course_idx,
                    cluster_id,
                    original_reward,
                    reward
                ])
        self._step_in_ep += 1'''

        self.nb_recommendations += 1
        terminated = self.nb_recommendations == self.k

        return observation, reward, terminated, False, info


class EvaluateCallback(BaseCallback):
    """Callback for evaluating the RL model during training.
    
    This callback evaluates the model's performance at regular intervals during training.
    It calculates the average number of applicable jobs across all learners and logs
    the results to a file.
    
    The evaluation process:
    1. Runs for each learner in the evaluation dataset
    2. Makes k course recommendations using the current policy
    3. Tracks the number of applicable jobs after each recommendation
    4. Calculates average performance across all learners
    
    Attributes:
        eval_env: Environment used for evaluation
        eval_freq (int): Frequency of evaluation in training steps
        all_results_filename (str): Path to save evaluation results
        mode (str): File opening mode ('w' for first write, 'a' for append)
    """
    
    def __init__(self, eval_env, eval_freq, all_results_filename, verbose=1):
        """Initialize the evaluation callback.
        
        Args:
            eval_env: Environment to use for evaluation
            eval_freq (int): Frequency of evaluation in training steps
            all_results_filename (str): Path to save evaluation results
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(EvaluateCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.all_results_filename = all_results_filename
        self.mode = "w"

    def _on_step(self):
        """Evaluate the model at regular intervals during training.
        
        This method:
        1. Evaluates the model every eval_freq steps
        2. Calculates average number of applicable jobs
        3. Logs results to file
        4. Prints progress information
        
        Returns:
            bool: True to continue training
        """
        # Only evaluate every 'eval_freq' training steps
        if self.n_calls % self.eval_freq == 0:
            time_start = process_time()  # Start timing the evaluation
            avg_jobs = 0  # Accumulator for average jobs across learners
            zero_rate = 0

            # Loop through each learner in the evaluation dataset
            for learner in self.eval_env.dataset.learners:
                self.eval_env.reset(options={"learner": learner})  # Reset environment with current learner
                done = False  # Flag to control evaluation episode
                tmp_avg_jobs = self.eval_env.unwrapped.get_info()["nb_applicable_jobs"]  # Initial jobs applicable without any recommendations

                # Run one full evaluation episode for the learner
                while not done:
                    obs = self.eval_env.unwrapped.get_obs()  # Get current observation (learner's skills)
                    if isinstance(self.model, MaskablePPO):
                        mask = self.eval_env.unwrapped.get_action_mask()
                        action, _state = self.model.predict(obs, deterministic=True, action_masks=mask)  # Predict action using current policy
                    else:
                        action, _state = self.model.predict(obs, deterministic=True)  # Predict action using current policy
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)  # Step in environment
                    done = terminated or truncated  # Properly compute done flag

                    # Only update if the recommendation was valid and use nb_applicable_jobs
                    if reward != -1:
                        tmp_avg_jobs = info["nb_applicable_jobs"]
                if tmp_avg_jobs == 0:
                    zero_rate += 1

                avg_jobs += tmp_avg_jobs  # Add learner's result to total

            time_end = process_time()  # End timing the evaluation

            # Log the result to the console
            print(
                f"Iteration {self.n_calls}. "
                f"Average jobs: {avg_jobs / len(self.eval_env.dataset.learners)} "
                f"Time: {time_end - time_start} "
                f"Zero rate: {zero_rate / len(self.eval_env.dataset.learners)}"
            )

            # Write evaluation result to file
            branch_dir = os.path.join(self.eval_env.dataset.config["results_path"], self.eval_env.dataset.config["branch_name"])
            data_dir = os.path.join(branch_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            with open(
                os.path.join(
                    data_dir,
                    self.all_results_filename,
                ),
                self.mode,  # 'w' for first time, 'a' for append afterward
            ) as f:
                f.write(
                    f"{self.n_calls} "
                    f"{avg_jobs / len(self.eval_env.dataset.learners)} "
                    f"{time_end - time_start}\n"
                )

            # After first write, switch mode to append for future evaluations
            if self.mode == "w":
                self.mode = "a"

        return True  # Returning True continues training