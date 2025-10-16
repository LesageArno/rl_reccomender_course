import os
import random

from typing import Callable, Optional

from numba import njit
import math

from time import process_time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

import matchings


class CourseRecEnv(gym.Env):
    """Course Recommendation Environment for Reinforcement Learning.

    This class implements a Gymnasium environment for course recommendations using
    reinforcement learning. The environment simulates the process of recommending
    courses to learners to help them acquire skills needed for jobs.

    The environment operates in two modes:
    1. Baseline: Uses number of applicable jobs as reward
    2. Skip-expertise: Uses a utility function that considers both skill acquisition
       and job applicability

    Observation Space:
        - Vector of length nb_skills representing learner's current skill levels
        - Each element is an integer in [0, max_level]
        - Shape: (nb_skills,)

    Action Space:
        - Discrete space of size nb_courses
        - Each action represents recommending a specific course
        - Range: [0, nb_courses-1]

    Attributes:
        dataset: Dataset object containing learners, jobs, and courses data
        nb_skills (int): Number of unique skills in the system
        mastery_levels (list): List of possible mastery levels for skills
        max_level (int): Maximum mastery level possible
        nb_courses (int): Number of available courses
        min_skills (int): Minimum number of skills a learner can have
        max_skills (int): Maximum number of skills a learner can have
        threshold (float): Minimum matching score required for job applicability
        k (int): Maximum number of course recommendations per learner
        baseline (bool): Whether to use baseline reward (True) or utility-based reward (False)
    """

    def __init__(self, dataset, threshold=0.8, k=3, baseline=False, method=1, feature="Usefulness-as-Rwd", beta1=0.5,
                 beta2=0.5, seed=42):
        """Initialize the course recommendation environment.

        Args:
            dataset: Dataset object containing the recommendation system data
            threshold (float, optional): Minimum matching score for job applicability. Defaults to 0.8.
            k (int, optional): Maximum number of course recommendations. Defaults to 3.
            baseline (bool, optional): Whether to use baseline reward. Defaults to False.
            feature (str, optional): Feature to use for reward. Defaults to "Usefulness-as-Rwd".
            beta1 (float, optional): Weight for number of applicable jobs in weighted reward. Defaults to 1.0.
            beta2 (float, optional): Weight for utility in weighted reward. Defaults to 1.0.
        """
        self.feature = feature
        self.baseline = baseline
        self.method = method
        self.beta1 = beta1
        self.beta2 = beta2
        self.dataset = dataset
        self.nb_skills = len(dataset.skills)  # 46 skills
        self.mastery_levels = [
            elem for elem in list(dataset.mastery_levels.values()) if elem > 0  # mastery level: [1,2,3,-1]
        ]
        self.max_level = max(self.mastery_levels)
        self.nb_courses = len(dataset.courses)  # 100 courses
        # get the minimum and maximum number of skills of the learners using np.nonzero
        self.min_skills = min(np.count_nonzero(self.dataset.learners, axis=1))  # 1
        self.max_skills = max(np.count_nonzero(self.dataset.learners, axis=1))  # 15
        self.threshold = threshold
        self.k = k
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        # The observation space is a vector of length nb_skills that represents the learner's skills.
        # The vector contains skill levels, where the minimum level is 0 and the maximum level is max_level (e.g., 3).
        # We cannot set the lower bound to -1 because negative values are not allowed in this Box space.
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_level, shape=(self.nb_skills + 1,), dtype=np.int32)

        # Define the action space for the environment.
        # This is a discrete space where each action corresponds to recommending a specific course.
        # The total number of possible actions is equal to the number of available courses (nb_courses = 100).
        # The agent will select an integer in [0, nb_courses - 1], representing the index of the recommended course.
        self.action_space = gym.spaces.Discrete(self.nb_courses)

    def get_obs(self):
        """Get the current observation of the environment.

        Returns:
            np.ndarray: Current learner's skill vector representing the state
        """
        step_left = np.array([(self.k - self.nb_recommendations) / self.k])

        obs = np.concatenate([self._agent_skills, step_left])
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
        if learner is None and options is not None:
            learner = options["learner"]

        if learner is not None:
            self._agent_skills = learner
        else:
            self._agent_skills = self.get_random_learner()
        self.nb_recommendations = 0
        self.rng = np.random.default_rng(self.seed)
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def calculate_course_metrics(self, learner, course):
        """Calculate N1, N2, N3 metrics for a course recommendation.

        These metrics evaluate the effectiveness of a course recommendation:
        - N1: Sum over all unachievable goals of the intersection between skills acquired after taking the course and missing skills for each goal
        - N2: Sum over all unachievable goals of the remaining missing skills after taking the course
        - N3: Number of skills provided by the course that are not in missing skills

        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]

        Returns:
            tuple: (N1, N2, N3) metrics
        """
        # Calculate skills after learning the course
        cons_skills = np.maximum(learner, course[1])
        cons_skills_set = set(np.nonzero(cons_skills > 0)[0])

        # Get skills provided by the course
        course_provided_skills = set(np.nonzero((course[1] - learner) > 0)[0])

        # Initialize N1 and N2
        N1 = 0
        N2 = 0

        # Calculate for each job
        for job_id in range(len(self.dataset.jobs)):
            # Get missing skills for this job before learning
            missing_skills = self.dataset.get_learner_missing_skills(learner, job_id)

            # Check if this job is in Ga (unachievable goals)
            if len(missing_skills) > 0:
                # Calculate N1: intersection of acquired skills and missing skills
                N1 += len(cons_skills_set.intersection(missing_skills))

                # Calculate N2: remaining missing skills after learning
                N2 += len(missing_skills - cons_skills_set)

        # Calculate N3: number of skills provided by the course that are not in any missing skills
        all_missing_skills = set()
        for job_id in range(len(self.dataset.jobs)):
            all_missing_skills.update(self.dataset.get_learner_missing_skills(learner, job_id))
        N3 = len(course_provided_skills - all_missing_skills)

        return N1, N2, N3

    def calculate_course_metrics_deficit(self, learner, course):
        """
        Calculate N1, N2, N3 metrics for a course recommendation (deficit-based version).

        In this version, missing skills are quantified as the number of mastery levels
        still lacking compared to job requirements.

        Definitions:
            - N1: Total reduction of deficits (in mastery levels) for all unachievable jobs.
            - N2: Total remaining deficits (in mastery levels) after taking the course,
                  for the same unachievable jobs.
            - N3: Total number of mastery levels gained from the course that are not
                  required by any unachievable job (i.e., overshoot or irrelevant gains).

        Args:
            learner (np.ndarray): Learner's skill vector (mastery levels, e.g., 0–3).
            course (np.ndarray): Course representation [required, provided], where
                                 course[1] is the vector of mastery levels provided.

        Returns:
            tuple: (N1, N2, N3) metrics as integers (measured in mastery levels).
        """
        #############################################################################################
        if self.dataset.config.get("use_numba", True):
            return _calc_metrics_deficit_numba(learner, course[1], self.dataset.jobs)
        ##############################################################################################

        # Skills after taking the course (target-level model)
        cons_skills = np.maximum(learner, course[1])

        # Initialize N1 and N2
        N1 = 0  # total deficit reduction across unachievable jobs
        N2 = 0  # total remaining deficits after the course

        # Track which skills are needed by at least one unachievable job
        needed = np.zeros(shape=(self.nb_skills,), dtype=bool)

        # Iterate over all jobs
        for job_id in range(len(self.dataset.jobs)):
            # Deficits before and after the course (per skill)
            n_missing_skills_before = np.clip(self.dataset.jobs[job_id] - learner, 0, None)
            n_missing_skills_after = np.clip(self.dataset.jobs[job_id] - cons_skills, 0, None)

            # Check if this job is in Ga (unachievable goals)
            if np.sum(n_missing_skills_before) > 0:
                # Mark skills that are needed for at least one unachievable job
                needed |= (n_missing_skills_before > 0)

                # N1: total deficit reduction (levels gained on missing skills)
                N1 += np.sum(n_missing_skills_before - n_missing_skills_after)

                # N2: total remaining deficits after the course
                N2 += np.sum(n_missing_skills_after)

        # Gains provided by the course per skill (in mastery levels)
        gains = np.maximum(0, cons_skills - learner)

        # N3: gains on skills not required by any unachievable job
        N3 = gains[~needed].sum()

        return N1, N2, N3

    def calculate_achievable_goals(self, learner, course):
        """Calculate the set of goals (jobs) that become achievable after taking a course.

        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]

        Returns:
            tuple: (initial_goals, new_goals) where:
                - initial_goals: Number of jobs applicable with current skills
                - new_goals: Number of jobs that become applicable after taking the course
        """
        # Calculate initial goals (jobs applicable with current skills)
        initial_goals = self.dataset.get_nb_applicable_jobs(learner, threshold=self.threshold)

        # Calculate skills after learning the course
        updated_skills = np.maximum(learner, course[1])

        # Calculate new goals (jobs applicable after learning the course)
        new_goals = self.dataset.get_nb_applicable_jobs(updated_skills, threshold=self.threshold)

        return initial_goals, new_goals

    def calculate_utility(self, learner, course, method=1):
        """Calculate the utility of a course recommendation.

        The utility function is defined as:
        U(φ) = 1/(|G|+1) * [|E(φ)| + N1(φ)/(N1(φ)+N2(φ)+(N3(φ)/(N3(φ)+1)))]

        where:
        - |G|: Number of jobs not applicable with initial skills
        - |E(φ)|: Number of new jobs that become applicable
        - N1: Number of missing skills resolved
        - N2: Number of remaining missing skills
        - N3: Number of additional skills provided

        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            method (int): Whether to use binary information or skills mastery deficit, default 1(mastery deficit)

        Returns:
            float: Utility value of the course recommendation
        """
        # Calculate N1, N2, N3 metrics
        if method == 0:
            N1, N2, N3 = self.calculate_course_metrics(learner, course)
        else:
            N1, N2, N3 = self.calculate_course_metrics_deficit(learner, course)

        # Calculate achievable goals
        initial_goals, new_goals = self.calculate_achievable_goals(learner, course)

        # Calculate |G|: number of jobs not applicable with initial skills
        total_jobs = len(self.dataset.jobs)
        Ga = total_jobs - initial_goals

        # Calculate |E(φ)|: number of new jobs that become applicable
        E_phi = new_goals - initial_goals

        # Calculate denominator for N1 fraction
        denominator = N1 + N2 + (N3 / (N3 + 1))
        if denominator == 0:  # Avoid division by zero
            N1_fraction = 0
        else:
            N1_fraction = N1 / denominator

        # Calculate U(φ)
        utility = (1 / (Ga + 1)) * (E_phi + N1_fraction)

        return utility

    def get_action_mask(self) -> np.ndarray:
        """
        Compute a boolean mask for valid course recommendations.

        A course is considered **invalid** if:
        - The learner does not meet the required skills (required_matching < threshold)
        - The learner already has all provided skills (provided_matching >= 1.0)

        Returns:
            np.ndarray: Boolean array (True = valid, False = invalid)
        """
        learner = self._agent_skills

        # === REQUIRED SKILLS MATCHING ===
        required_skills = self.dataset.courses[:, 0, :].astype(float)  # [num_courses, num_skills]
        has_requirement = required_skills > 0  # mask of required skills
        required_safe = np.where(has_requirement, required_skills, 1.0)  # avoid divide by 0

        # Compute fractional match per skill: min(learner_level, required_level) / required_level
        required_fraction = np.minimum(learner, required_skills) / required_safe
        required_fraction[~has_requirement] = 0.0  # ignore non-required skills

        # Aggregate to average matching per course
        required_sum = required_fraction.sum(axis=1)
        required_count = has_requirement.sum(axis=1)
        required_matching = np.divide(
            required_sum, required_count, out=np.ones_like(required_sum), where=(required_count > 0)
        )
        # If no prerequisites, matching = 1.0 (course always valid in that regard)
        required_matching[required_count == 0] = 1.0

        # === PROVIDED SKILLS MATCHING ===
        provided_skills = self.dataset.courses[:, 1, :].astype(float)
        has_provided = provided_skills > 0
        provided_safe = np.where(has_provided, provided_skills, 1.0)

        # Fractional overlap: min(learner_level, provided_level) / provided_level
        provided_fraction = np.minimum(learner, provided_skills) / provided_safe
        provided_fraction[~has_provided] = 0.0

        provided_sum = provided_fraction.sum(axis=1)
        provided_count = has_provided.sum(axis=1)
        provided_matching = np.divide(
            provided_sum, provided_count, out=np.zeros_like(provided_sum), where=(provided_count > 0)
        )
        # If no provided skills, treat as 0.0
        provided_matching[provided_count == 0] = 0.0

        # === VALIDITY RULE ===
        valid_courses = (required_matching >= self.threshold) & (provided_matching < 1.0)

        # Defensive fallback (if all invalid, allow first one)
        if not valid_courses.any():
            valid_courses[0] = True

        return valid_courses
        '''learner = self._agent_skills
        valid = np.ones(self.dataset.courses.shape[0], dtype=bool)

        for i, course in enumerate(self.dataset.courses):
            req = matchings.learner_course_required_matching(learner, course)
            prov = matchings.learner_course_provided_matching(learner, course)
            if prov >= 1.0 or req < self.threshold:
                valid[i] = False

        if not valid.any():  # defensive fallback
            valid[0] = True
        return valid'''

    def _sample_mastery_outcome(self, base_levels):
        """
        base_levels: course level target {1,2,3}
        :returns updated levels resulting from potential overlearning.
        """
        base_levels = np.asarray(base_levels, dtype=int)
        p_double = 0.008
        p_over = 0.05

        mask = base_levels > 0

        k = np.sum(mask)
        p_double /= k
        p_over /= k

        # Independent Bernoulli for +2 and +1
        # Note: if +2 triggers, +1 is not applied (higher jump has priority)
        jump2 = self.rng.random(base_levels.shape) < p_double
        jump1 = (self.rng.random(base_levels.shape) < p_over) & (~jump2)

        outcome = base_levels + jump1.astype(int) + (2 * jump2.astype(int))
        outcome = np.clip(outcome, 0, 3)  # cap levels at 3
        outcome[~mask] = 0

        return outcome

    def step(self, action):
        """Execute one step in the environment.

        This method:
        1. Recommends a course based on the action
        2. Updates the learner's skills if the course is valid
        3. Calculates the reward based on the selected mode:
           - Baseline: Number of applicable jobs
           - Usefulness-of-info-as-Rwd: Utility function value
           - Weighted-Usefulness-of-info-as-Rwd: Number of applicable jobs + Utility
        4. Checks if the episode should terminate

        Args:
            action (int): Index of the course to recommend

        Returns:
            tuple: (observation, reward, terminated, truncated, info) where:
                - observation: Updated learner's skill vector
                - reward: Reward value based on the selected mode
                - terminated: Whether the episode is done
                - truncated: Whether the episode was truncated
                - info: Additional information about the step
        """
        course = self.dataset.courses[action]
        learner = self._agent_skills

        # Skip-expertise case: use new metrics and utility
        required_matching = matchings.learner_course_required_matching(learner, course)
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        if provided_matching == 1.0 or required_matching < self.threshold:
            observation = self.get_obs()
            reward = -1
            terminated = True
            info = self.get_info()
            return observation, reward, terminated, False, info

        if self.baseline:  # baseline model
            self._agent_skills = np.maximum(self._agent_skills, course[1])
            observation = self.get_obs()
            info = self.get_info()
            reward = info["nb_applicable_jobs"]
        else:  # No-Mastery-Levels Models
            # Calculate Usefulness-of-info-as-Rwd
            utility = self.calculate_utility(learner, course, self.method)

            # learned_course = self._sample_mastery_outcome(course[1])

            self._agent_skills = np.maximum(self._agent_skills, course[1])  # learned_course)
            observation = self.get_obs()
            info = self.get_info()
            info["utility"] = utility

            if self.feature == "Usefulness-as-Rwd":
                reward = info["utility"]  # Use utility as reward
            elif self.feature == "Weighted-Usefulness-as-Rwd":
                reward = self.beta1 * info["nb_applicable_jobs"] + self.beta2 * info[
                    "utility"]  # Combine both metrics with weights
            else:
                raise ValueError(f"Unknown feature type: {self.feature}")

        self.nb_recommendations += 1
        terminated = self.nb_recommendations == self.k

        return observation, reward, terminated, False, info


class EvaluateCallback(BaseCallback):
    """Callback for evaluating the RL model during training.

    This callback evaluates the model's performance at regular intervals during training.
    It calculates the average number of applicable jobs across all learners and logs
    the results_k2 to a file.

    Attributes:
        eval_env: Environment used for evaluation
        eval_freq (int): Frequency of evaluation in training steps
        all_results_filename (str): Path to save evaluation results_k2
        mode (str): File opening mode ('w' for first write, 'a' for append)
    """

    def __init__(self, eval_env, eval_freq, all_results_filename, verbose=1):
        """Initialize the evaluation callback.

        Args:
            eval_env: Environment to use for evaluation
            eval_freq (int): Frequency of evaluation in training steps
            all_results_filename (str): Path to save evaluation results_k2
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(EvaluateCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.all_results_filename = all_results_filename
        self.mode = "w"

        self._anneal_done = False  # run-once switch

        # --- hook for Optuna/ASHA ---
        self.report_fn: Optional[Callable[[int, float], bool]] = None
        self.was_pruned: bool = False
        self.last_avg_jobs: Optional[float] = None
        self._eval_calls: int = 0

    def cosine_anneal(self, value_start, value_end, step, total_steps, start_frac=0.7):

        start_step = total_steps * start_frac
        if step <= start_step:
            return value_start
        progress = min((step - start_step) / (total_steps - start_step), 1.0)
        weight = 0.5 * (1 + math.cos(math.pi * progress))
        return value_end + (value_start - value_end) * weight

    def _on_step(self):
        """Evaluate the model at regular intervals during training.

        This method:
        1. Evaluates the model every eval_freq steps
        2. Calculates average number of applicable jobs
        3. Logs results_k2 to file
        4. Prints progress information

        Returns:
            bool: True to continue training
        """
        # Only evaluate every 'eval_freq' training steps
        if self.n_calls % self.eval_freq == 0:
            total_steps = getattr(self.model, "_total_timesteps", 500_000)
            initial_ent = getattr(self.model, 'ent_coef', 0.025)
            initial_clip = getattr(self.model, 'clip_range', 0.25)
            if callable(initial_clip):
                initial_clip = initial_clip(1.0)
            ent = self.cosine_anneal(initial_ent, 0.006, self.n_calls, total_steps)
            clip = self.cosine_anneal(initial_clip, 0.12, self.n_calls, total_steps)

            if hasattr(self.model, "ent_coef"):
                self.model.ent_coef = ent
            if hasattr(self.model, "params"):
                self.model.params["ent_coef"] = ent

            if hasattr(self.model, "clip_range"):
                self.model.clip_range = lambda _: clip
            if hasattr(self.model, "params"):
                self.model.params["clip_range"] = clip
            time_start = process_time()  # Start timing the evaluation
            avg_jobs = 0  # Accumulator for average jobs across learners

            # Loop through each learner in the evaluation dataset
            for learner in self.eval_env.unwrapped.dataset.learners:
                self.eval_env.reset(options={"learner": learner})  # Reset environment with current learner
                done = False  # Flag to control evaluation episode
                tmp_avg_jobs = self.eval_env.unwrapped.get_info()[
                    "nb_applicable_jobs"]  # Initial jobs applicable without any recommendations

                # Run one full evaluation episode for the learner
                while not done:
                    obs = self.eval_env.unwrapped.get_obs()
                    if isinstance(self.model, MaskablePPO):
                        mask = self.eval_env.unwrapped.get_action_mask()
                        action, _state = self.model.predict(obs, action_masks=mask, deterministic=True)
                    else:
                        action, _state = self.model.predict(obs,
                                                            deterministic=True)  # Predict action using current policy
                    # obs = self.eval_env.get_obs()  # Get current observation (learner's skills)

                    obs, reward, terminated, truncated, info = self.eval_env.step(action)  # Step in environment
                    done = terminated or truncated  # Properly compute done flag

                    # Only update if the recommendation was valid and use nb_applicable_jobs
                    if reward != -1:
                        tmp_avg_jobs = info["nb_applicable_jobs"]

                avg_jobs += tmp_avg_jobs  # Add learner's result to total

            time_end = process_time()  # End timing the evaluation

            avg = avg_jobs / len(self.eval_env.unwrapped.dataset.learners)

            # Log the result to the console
            print(
                f"Iteration {self.n_calls}. "
                f"Average jobs: {avg} "
                f"Time: {time_end - time_start}"
            )

            results_dir = os.path.join(
                f"{self.eval_env.unwrapped.dataset.config['results_path']}{self.eval_env.unwrapped.dataset.config['k']}/seed{self.eval_env.unwrapped.dataset.config['seed']}"
            )

            # Create directory if it does not already exist
            os.makedirs(results_dir, exist_ok=True)

            file_path = os.path.join(results_dir, self.all_results_filename)

            # Write evaluation result to file
            with open(file_path, self.mode) as f:  # 'w' for first time, 'a' for append afterward
                f.write(
                    f"{self.n_calls} "
                    f"{avg_jobs / len(self.eval_env.unwrapped.dataset.learners)} "
                    f"{time_end - time_start}\n"
                )

            # After first write, switch mode to append for future evaluations
            if self.mode == "w":
                self.mode = "a"

            self.last_avg_jobs = float(avg)
            self._eval_calls += 1

            if callable(self.report_fn):
                keep = bool(self.report_fn(self._eval_calls, self.last_avg_jobs))
                if not keep:
                    self.was_pruned = True
                    return False  # interrupts learn()

        return True  # Returning True continues training


@njit(cache=True)
def _calc_metrics_deficit_numba(learner: np.ndarray,
                                course_provided: np.ndarray,
                                jobs: np.ndarray) -> tuple:
    """
    Numba-compiled helper replicating calculate_course_metrics_deficit.
    Returns (N1, N2, N3) as ints.
    """

    num_jobs, num_skills = jobs.shape

    # After taking the course (max level per skill)
    cons_skills = np.empty_like(learner)
    for s in range(num_skills):
        l = learner[s]
        c = course_provided[s]
        cons_skills[s] = l if l >= c else c

    N1 = 0.0  # total deficit reduction
    N2 = 0.0  # remaining deficits
    needed = np.zeros(num_skills, dtype=np.bool_)

    # --- iterate over all jobs ---
    for j in range(num_jobs):
        denom_before = 0.0
        denom_after = 0.0
        missing_before = np.zeros(num_skills)
        missing_after = np.zeros(num_skills)
        has_deficit = False

        for s in range(num_skills):
            job_req = jobs[j, s]
            if job_req > 0.0:
                lv = learner[s]
                cs = cons_skills[s]
                diff_before = job_req - lv
                diff_after = job_req - cs

                if diff_before > 0.0:
                    has_deficit = True
                    needed[s] = True

                # clamp negative values to 0
                if diff_before < 0.0:
                    diff_before = 0.0
                if diff_after < 0.0:
                    diff_after = 0.0

                missing_before[s] = diff_before
                missing_after[s] = diff_after

        if has_deficit:
            # N1: total deficit reduction
            for s in range(num_skills):
                N1 += (missing_before[s] - missing_after[s])
                N2 += missing_after[s]

    # --- N3: total gains on irrelevant skills ---
    N3 = 0.0
    for s in range(num_skills):
        gain = cons_skills[s] - learner[s]
        if gain > 0.0 and not needed[s]:
            N3 += gain

    return N1, N2, N3
