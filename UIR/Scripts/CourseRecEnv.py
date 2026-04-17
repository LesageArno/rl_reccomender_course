import os
import random

from typing import Callable, Optional

from numba import njit
import math

import time
from time import process_time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from UIR.Scripts import matchings


class CourseRecEnv(gym.Env):
    """Course Recommendation Environment for Reinforcement Learning.

    This class implements a Gymnasium environment for course recommendations using
    reinforcement learning. The environment simulates the process of recommending
    courses to learners to help them acquire skills needed for jobs.

    The environment operates in two modes:
    1. Baseline: Uses Employability as reward
    2. UIR: Uses a utility function that considers both skill acquisition
       and job applicability

    Observation Space:
        2 types:
        without preferences:
        - Vector of length nb_skills representing learner's current skill levels
        - Each element is an integer in [0, max_level]
        - Shape: (nb_skills,)
        with preferences:
        - Vector of length nb_skills * 3 representing learner's current skill levels, wanted skills, avoided skills
        - Each element is an integer in [0, max_level]
        - Shape: (nb_skills * 3,)

    Action Space:
        - Discrete space of size nb_courses
        - Each action represents recommending a specific course
        - Range: [0, nb_courses-1]

    Attributes:
        dataset: Dataset object containing learners, jobs, and courses data
        nb_skills (int): Number of unique skills considered in the system
        mastery_levels (list): List of possible mastery levels for skills
        max_level (int): Maximum mastery level possible
        nb_courses (int): Number of available courses
        min_skills (int): Minimum number of skills a learner can have
        max_skills (int): Maximum number of skills a learner can have
        threshold (float): Minimum matching score required for job applicability
        k (int): Maximum number of course recommendations per learner
        baseline (bool): Whether to use Employability as reward (True) or ```UIR/EUIR``` as reward (False)
    """

    def __init__(self, dataset, config, k=3, fuzzyMode=False):
        """Initialize the course recommendation environment.

        Args:
            dataset: Dataset object containing the recommendation system data
            config (dict): Configuration parameters for the environment
            k (int, optional): Maximum number of course recommendations. Defaults to 3.
        """
        self.config = config
        self.fuzzyMode = fuzzyMode
        self.feature = config.get("feature", "UIR")
        self.baseline = self.feature == "Employability"
        self.method = config.get("method", 1)
        self.threshold = config.get("threshold", 0.8)

        
        
        self.dataset = dataset
        self.jobs = self.dataset.jobs  # None = usa tutti i job, altrimenti solo un sottoinsieme
        
        self.courses = dataset.courses
        ########################### USEFUL FOR ACTION MASKING
        self._req_skills = self.courses[:, 0, :].astype(np.float32)
        self._req_has = self._req_skills > 0
        self._req_safe = np.where(self._req_has, self._req_skills, 1.0).astype(np.float32)
        self._req_count = self._req_has.sum(axis=1).astype(np.int32)

        self._prov_skills = self.courses[:, 1, :].astype(np.float32)
        self._prov_has = self._prov_skills > 0
        self._prov_safe = np.where(self._prov_has, self._prov_skills, 1.0).astype(np.float32)
        self._prov_count = self._prov_has.sum(axis=1).astype(np.int32)
        ############################

        self.extra_invalid_actions = None
        self.nb_skills = len(dataset.skills)  # 46 skills
        self.mastery_levels = [
            elem for elem in list(dataset.mastery_levels.values()) if elem > 0  # mastery level: [1,2,3,-1]
        ]
        self.max_level = max(self.mastery_levels)
        self.nb_courses = len(dataset.courses)  # 100 courses
        # get the minimum and maximum number of skills of the learners using np.nonzero
        self.min_skills = min(np.count_nonzero(self.dataset.learners, axis=1))  # 1
        self.max_skills = max(np.count_nonzero(self.dataset.learners, axis=1))  # 15
        self.k = k
        self.seed = config.get("seed", 42)
        self.rng = np.random.default_rng(seed=self.seed)

        # The observation space is either:
        # - a vector of length nb_skills that represents the learner's skills.
        # or
        # - a vector of lenght 3 * nb_skills containing leaner's skills and preferences
        # Both containing also the number of steps left
        # The vector contains skill levels, where the minimum level is 0 and the maximum level is 3.
        # We cannot set the lower bound to -1 because negative values are not allowed in this Box space.
        if self.config.get("use_preference", True):
            high = np.concatenate([
                np.full(self.nb_skills, self.max_level, dtype=np.int32),  # skills
                np.ones(self.nb_skills, dtype=np.int32),                  # want
                np.ones(self.nb_skills, dtype=np.int32),                  # avoid
                np.array([self.k], dtype=np.int32),                       # step_left
            ])
            low = np.zeros_like(high)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)
        else: 
            high = np.concatenate([
                np.full(self.nb_skills, self.max_level, dtype=np.int32),  # skills
                np.array([self.k], dtype=np.int32),                       # step_left
            ])
            low = np.zeros_like(high)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)


        self._want = np.zeros(self.nb_skills, dtype=np.int32)
        self._avoid = np.zeros(self.nb_skills, dtype=np.int32)
        self.jobs_goal = self.jobs  # default: all


        # Define the action space for the environment.
        # This is a discrete space where each action corresponds to recommending a specific course.
        # The total number of possible actions is equal to the number of available courses (nb_courses = 100).
        # The agent will select an integer in [0, nb_courses - 1], representing the index of the recommended course.
        self.action_space = gym.spaces.Discrete(self.nb_courses)

    def set_extra_invalid_actions(self, invalid_actions_ids):
        """
        invalid_actions_ids: ids of avoided courses (es. [0, 3, 5, ...])
        """
        if invalid_actions_ids is None:
            self.extra_invalid_actions = None
            return

        mask = np.zeros(self.nb_courses, dtype=bool)
        for idx in invalid_actions_ids:
            if 0 <= idx < self.nb_courses:
                mask[idx] = True  

        self.extra_invalid_actions = mask

    def get_obs(self):
        """Get the current observation of the environment.

        Returns:
            np.ndarray: Current learner's skill vector representing the state
        """
        step_left = np.array([self.k - self.nb_recommendations], dtype=np.int32)
        
        if self.config.get("use_preference", True):
            obs = np.concatenate([self._agent_skills, self._want, self._avoid, step_left])
        else:
            obs = np.concatenate([self._agent_skills, step_left])
        return obs

    def get_info(self):
        """Get additional information about the current state.

        Returns:
            dict: Dictionary containing information about the current state, including:
                nb_applicable_jobs: Number of applicable job_goals for the current learner
                goal_gap_total: Total goal gap for the current learner
                pref_coverage: Preference coverage for the current learner
                total_skill_levels_required: Total number of skill levels required for the current learner
                skills_required_unique: Number of skills required for the current learner
                skills_fully_covered: Number of skills fully covered
                skills_missing_unique: Number of skills missing 
        """
        learner = self._agent_skills

        # Employability on goal set
        employability_goal = self.dataset.get_nb_applicable_jobs(
            learner, threshold=self.threshold, jobs=self.jobs_goal
        )

        # Total skill gap on goal set (sum of missing levels)
        # Aggregate levels across all jobs and cap at 3
        required_levels = self.jobs_goal.sum(axis=0).clip(0, 3)

        # Calculate covered levels based on the learner's current state
        covered_levels = np.minimum(required_levels, learner)

        # levels_missing: remaining levels to reach the goal per skill
        levels_missing = (required_levels - covered_levels)

        # Total gap is the sum of levels still missing to reach the required profile
        goal_gap_total = int(levels_missing.sum())
        
        # Total number of skill levels involved in the goal
        total_skill_levels_required = int(required_levels.sum())

        # Count of unique skills involved in the goal
        skills_required_unique = int((required_levels > 0).sum())

        # Count of skills where learner level >= required level
        skills_fully_covered = int(((required_levels > 0) & (learner >= required_levels)).sum())

        # Count of skills that still need at least one level
        skills_missing_unique = int((levels_missing > 0).sum())

        # Preference coverage (wanted skills improved during episode)
        want_n = int(self._want.sum())
        pref_coverage = float(self.covered_want.sum() / want_n) if want_n > 0 else 0.0

        return {
            "nb_applicable_jobs": employability_goal,
            "goal_gap_total": goal_gap_total,
            "pref_coverage": pref_coverage,
            "total_skill_levels_required": total_skill_levels_required,
            "skills_required_unique": skills_required_unique,
            "skills_fully_covered": skills_fully_covered,
            "skills_missing_unique": skills_missing_unique
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
        skill_indices = self.rng.choice(self.nb_skills, size=n_skills, replace=False)

        # Assign random mastery levels to these skills, levels can repeat
        initial_skills[skill_indices] = self.rng.choice(
            self.mastery_levels, size=n_skills, replace=True
        )
        return initial_skills
    
    def _eval_want_avoid(self, learner, learner_idx, base_seed=123):
        """
        Deterministic eval preferences:
        same learner_idx + same base_seed => same want/avoid forever.
        Does NOT use self.rng.
        """
        rng = np.random.default_rng(base_seed + int(learner_idx))

        # WANT: 1–4 uniform
        want = np.zeros(self.nb_skills, dtype=np.int32)
        n_want = rng.integers(1, 5)
        chosen_w = rng.choice(self.nb_skills, size=n_want, replace=False)
        want[chosen_w] = 1

        # AVOID: 0–2 biased toward 0, no overlap with WANT
        avoid = np.zeros(self.nb_skills, dtype=np.int32)
        n_avoid = rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15])                    
        if n_avoid > 0:
            all_skills = np.arange(self.nb_skills)
            candidates = np.setdiff1d(all_skills, chosen_w)
            if candidates.size > 0:
                chosen_a = rng.choice(candidates, size=min(n_avoid, candidates.size), replace=False)
                avoid[chosen_a] = 1

        return want, avoid
    
    def _sample_want_random(self):
        """
        Sample WANT skills uniformly at random.
        - Number of skills: 1 to 4
        Output: binary vector of length nb_skills
        """
        want = np.zeros(self.nb_skills, dtype=np.int32)

        # number of wanted skills
        n_want = self.rng.integers(1, 5)  

        # sample skills uniformly from the whole skill space
        chosen = self.rng.choice(self.nb_skills, size=n_want, replace=False) 
        want[chosen] = 1
        #want[39] = 1

        return want

    def _sample_avoid_random(self):
        """
        Sample AVOID skills with a bias toward having none.
        - Number of skills: 0 to 2
        - More often 0 than >0
        - No overlap with WANT (self._want)
        Output: binary vector of length nb_skills
        """
        avoid = np.zeros(self.nb_skills, dtype=np.int32)

        # biased distribution: mostly no avoid preferences
        n_avoid = self.rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15])           #change to 0 1 2
        if n_avoid == 0:
            return avoid

        # candidate skills = all skills except WANT
        all_skills = np.arange(self.nb_skills)
        candidate_skills = np.setdiff1d(all_skills, np.nonzero(self._want)[0])

        if len(candidate_skills) == 0:
            return avoid

        chosen = self.rng.choice(
            candidate_skills,
            size=min(n_avoid, len(candidate_skills)),
            replace=False
        )
        avoid[chosen] = 1

        return avoid

    def _build_goal_jobs_W1_hard(self):
        """
        Build goal-job mask G_a using W1-hard:
        - keep jobs that require at least one WANT skill
        - exclude jobs that require any AVOID skill

        Returns:
            np.ndarray(bool): mask of shape (nb_jobs,)
        Fallback:
            if empty, relax AVOID (keep only the WANT constraint).
        """
        jobs_req = self.jobs  # shape: [J, S], values >0 mean "required"

        want_idx = np.nonzero(self._want)[0]
        avoid_idx = np.nonzero(self._avoid)[0]

        # --- WANT condition: job requires at least one wanted skill ---
        if want_idx.size == 0:
            has_want = np.ones(jobs_req.shape[0], dtype=bool)  # no want => don't filter by want
        else:
            has_want = (jobs_req[:, want_idx] > 0).any(axis=1)

        # --- AVOID condition: job requires none of the avoided skills ---
        if avoid_idx.size == 0:
            has_avoid = np.zeros(jobs_req.shape[0], dtype=bool)
        else:
            has_avoid = (jobs_req[:, avoid_idx] > 0).any(axis=1)

        mask = has_want & (~has_avoid)

        # Defensive fallback: if empty, relax avoid (still goal-conditioned by want)
        if not mask.any():
            mask = has_want.copy()
            if not mask.any():
                mask[:] = True  # last resort: use all jobs

        return jobs_req[mask]

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
        #self.rng = np.random.default_rng(self.seed)

        want = None
        avoid = None
        if options is not None:
            want = options.get("want", None)
            avoid = options.get("avoid", None)

        self._want = want if want is not None else self._sample_want_random()
        self._avoid = avoid if avoid is not None else self._sample_avoid_random()

        self.jobs_goal = self._build_goal_jobs_W1_hard()
        self.jobs_goal = np.ascontiguousarray(self.jobs_goal)  # For Numba in order to work faster
        #print(len(self.jobs_goal))
        self.covered_want = np.zeros(self.nb_skills, dtype=bool)

        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def calculate_course_metrics(self, learner: np.ndarray, course: np.ndarray) -> tuple:
        """
        Calculate (Nr, Nm, Nnr) for a course recommendation (THRESHOLD-BASED, mastery-aware).

        Mastery-aware interpretation (per job g and skill s):
        - A skill is missing before the course if: req_g[s] > learner[s]
        - After the course, the learner's level becomes: cons[s] = max(learner[s], provided[s])
        - A missing skill is considered "covered" for job g if: cons[s] >= req_g[s]

        Metrics:
        - Nr: Sum over all unachievable jobs (having at least one missing skill BEFORE the course)
                of the number of missing skills that become covered AFTER the course.
        - Nm: Sum over the same jobs of the number of missing skills that remain uncovered AFTER the course.
        - Nnr: Number of skills improved by the course (provided > learner) that are not missing
                for ANY job (union of missing skills BEFORE the course).

        Args:
            learner (np.ndarray): Learner skill vector (mastery levels).
            course (np.ndarray): Course representation [required, provided], where course[1] is provided.

        Returns:
            tuple: (Nr, Nm, Nnr) as integers.
        """
        ##################################################################
        #                            NUMBA                               #
        ##################################################################
        if self.config.get("use_numba", True):
            return _calc_metrics_threshold_mastery_numba(learner, course[1], self.jobs_goal)
        ##################################################################
        ##################################################################

        # Learner levels after taking the course (target-level model)
        cons_skills = np.maximum(learner, course[1])

        # Skills strictly improved by this course (used for Nnr)
        provided_new = course[1] > learner  # boolean mask

        Nr = 0
        Nm = 0

        # --- Nr and Nm: job-wise contributions over "unachievable" jobs (missing_before.any()) ---
        for job_id in range(len(self.jobs_goal)):
            req = self.jobs_goal[job_id]

            # Missing skills BEFORE course for this job (boolean mask)
            missing_before = self.dataset.get_learner_missing_skills(learner, job_id, jobs = self.jobs_goal)

            # "Unachievable" ⇔ at least one missing skill before course
            if missing_before.any():
                # A missing skill is covered if AFTER the course cons >= req
                covered = missing_before & (cons_skills >= req)

                Nr += int(covered.sum())
                Nm += int(missing_before.sum() - covered.sum())

        # --- Nnr: improved skills that are not missing for ANY job BEFORE the course ---
        # Union of missing skills across all jobs (boolean OR over jobs)
        all_missing_any = (self.jobs_goal > learner).any(axis=0)

        Nnr = int((provided_new & ~all_missing_any).sum())

        return Nr, Nm, Nnr

    def calculate_course_metrics_gap(self, learner, course):
        """
        Calculate Nr, Nm, Nnr metrics for a course recommendation (GA-BASED-METHOD).

        In this version, missing skills are quantified as the number of mastery levels
        still lacking compared to job requirements.

        Definitions:
            - Nr: Total reduction of deficits (in mastery levels) for all unachievable jobs.
            - Nm: Total remaining deficits (in mastery levels) after taking the course,
                  for the same unachievable jobs.
            - Nnr: Total number of mastery levels gained from the course that are not
                  required by any unachievable job (i.e., overshoot or irrelevant gains).

        Args:
            learner (np.ndarray): Learner's skill vector (mastery levels, e.g., 0–3).
            course (np.ndarray): Course representation [required, provided], where
                                 course[1] is the vector of mastery levels provided.

        Returns:
            tuple: (Nr, Nm, Nnr) metrics as integers (measured in mastery levels).
        """
        #############################################################################################
        if self.config.get("use_numba", True):
            return _calc_metrics_deficit_numba(learner, course[1], self.jobs_goal)
        ##############################################################################################

        # Skills after taking the course (target-level model)
        cons_skills = np.maximum(learner, course[1])

        # Initialize Nr and Nm
        Nr = 0  # total deficit reduction across unachievable jobs
        Nm = 0  # total remaining deficits after the course

        # Track which skills are needed by at least one unachievable job
        needed = np.zeros(shape=(self.nb_skills,), dtype=bool)

        # Iterate over all jobs
        for job_id in range(len(self.jobs_goal)):
            # Deficits before and after the course (per skill)
            n_missing_skills_before = np.clip(self.jobs_goal[job_id] - learner, 0, None)
            n_missing_skills_after = np.clip(self.jobs_goal[job_id] - cons_skills, 0, None)

            # Check if this job is in Ga (unachievable goals)
            if np.sum(n_missing_skills_before) > 0:
                # Mark skills that are needed for at least one unachievable job
                needed |= (n_missing_skills_before > 0)

                # Nr: total deficit reduction (levels gained on missing skills)
                Nr += np.sum(n_missing_skills_before - n_missing_skills_after)

                # Nm: total remaining deficits after the course
                Nm += np.sum(n_missing_skills_after)

        # Gains provided by the course per skill (in mastery levels)
        gains = np.maximum(0, cons_skills - learner)

        # Nnr: gains on skills not required by any unachievable job
        Nnr = gains[~needed].sum()

        return Nr, Nm, Nnr

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
        initial_goals = self.dataset.get_nb_applicable_jobs(learner, threshold=self.threshold, jobs=self.jobs_goal)

        # Calculate skills after learning the course
        updated_skills = np.maximum(learner, course[1])

        # Calculate new goals (jobs applicable after learning the course)
        new_goals = self.dataset.get_nb_applicable_jobs(updated_skills, threshold=self.threshold, jobs=self.jobs_goal)

        return initial_goals, new_goals

    def calculate_utility(self, learner, course, method=1):
        """Calculate the utility of a course recommendation.

        The utility function is defined as:
        U(φ) = 1/(|G|+1) * [|E(φ)| + Nr(φ)/(Nr(φ)+Nm(φ)+(Nnr(φ)/(Nnr(φ)+1)))]

        where:
        - |G|: Number of jobs not applicable with initial skills
        - |E(φ)|: Number of new jobs that become applicable
        - Nr: Number of missing skills resolved
        - Nm: Number of remaining missing skills
        - Nnr: Number of additional skills provided

        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            method (int): Whether to use binary information or skills mastery deficit, default 1(mastery deficit)

        Returns:
            float: Utility value of the course recommendation
        """
        # Calculate Nr, Nm, Nnr metrics
        if method == 0:
            Nr, Nm, Nnr = self.calculate_course_metrics(learner, course)
        else:
            Nr, Nm, Nnr = self.calculate_course_metrics_gap(learner, course)

        # Calculate achievable goals
        initial_goals, new_goals = self.calculate_achievable_goals(learner, course)

        # Calculate |G|: number of jobs not applicable with initial skills
        total_jobs = len(self.jobs_goal)
        Ga = total_jobs - initial_goals

        # Calculate |E(φ)|: number of new jobs that become applicable
        E_phi = new_goals - initial_goals

        # Calculate denominator for Nr fraction
        denominator = Nr + Nm + (Nnr / (Nnr + 1))
        if denominator == 0:  # Avoid division by zero
            Nr_fraction = 0
        else:
            Nr_fraction = Nr / denominator

        # Calculate U(φ)
        utility = (1 / (Ga + 1)) * (E_phi + Nr_fraction)

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
        #required_skills = self.dataset.courses[:, 0, :].astype(float)  # [num_courses, num_skills]
        #has_requirement = required_skills > 0  # mask of required skills
        #required_safe = np.where(has_requirement, required_skills, 1.0)  # avoid divide by 0

        # Compute fractional match per skill: min(learner_level, required_level) / required_level
        required_fraction = np.minimum(learner, self._req_skills) / self._req_safe
        required_fraction[~self._req_has] = 0.0  # ignore non-required skills

        # Aggregate to average matching per course
        required_sum = required_fraction.sum(axis=1)
        #required_count = self._req_has.sum(axis=1)
        required_matching = np.divide(
            required_sum, self._req_count, out=np.ones_like(required_sum), where=(self._req_count > 0)
        )
        # If no prerequisites, matching = 1.0 (course always valid in that regard)
        required_matching[self._req_count == 0] = 1.0

        # === PROVIDED SKILLS MATCHING ===
        #provided_skills = self.dataset.courses[:, 1, :].astype(float)
        #has_provided = provided_skills > 0
        #provided_safe = np.where(has_provided, provided_skills, 1.0)

        # Fractional overlap: min(learner_level, provided_level) / provided_level
        provided_fraction = np.minimum(learner, self._prov_skills) / self._prov_safe
        provided_fraction[~self._prov_has] = 0.0

        provided_sum = provided_fraction.sum(axis=1)
        #provided_count = has_provided.sum(axis=1)
        provided_matching = np.divide(
            provided_sum, self._prov_count, out=np.zeros_like(provided_sum), where=(self._prov_count > 0)
        )
        # If no provided skills, treat as 0.0
        provided_matching[self._prov_count == 0] = 0.0

        # === VALIDITY RULE ===
        valid_courses = (required_matching >= self.threshold) & (provided_matching < 1.0)

        if self.extra_invalid_actions is not None:
            valid_courses = valid_courses & (~self.extra_invalid_actions)

        # Defensive fallback (if all invalid, allow first one)
        if not valid_courses.any():
            # pick one deterministic action to keep distribution valid
            valid_courses[:] = False
            valid_courses[0] = True

        return valid_courses

    def _sample_mastery_outcome(self, base_levels):
        """
        This function is useful if want to address potentially overlearning from a course.

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
        outcome = np.clip(outcome, 0, 3)  
        outcome[~mask] = 0

        return outcome

    def step(self, action):
        """Execute one step in the environment.

        This method:
        1. Recommends a course based on the action
        2. Updates the learner's skills if the course is valid
        3. Calculates the reward based on the selected mode:
           - Baseline: Employability as rwd model
           - UIR: Utility function value as rwd model
           - EUIR: (normalized Employability) + Utility function value as rwd model
        4. Checks if the episode should terminate

        Args:
            action (int): Index of the course to recommend

        Returns:
            tuple: (observation, reward, terminated, truncated, info) where:
                - observation: Updated learner's skill profile
                - reward: Reward value based on the selected mode
                - terminated: Whether the episode is done
                - truncated: Whether the episode was truncated
                - info: Additional information about the step
        """
        course = self.courses[action]
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

        if self.baseline:  # Employability as rwd model
            self._agent_skills = np.maximum(self._agent_skills, course[1])
            
            # Update preferences covered skills if any
            improved = (self._agent_skills > learner)
            self.covered_want |= improved & (self._want.astype(bool))
            
            observation = self.get_obs()
            info = self.get_info()
            reward = info["nb_applicable_jobs"]
        else:  # UIR-models
            # Calculate Usefulness-of-info-as-Rwd
            utility = self.calculate_utility(learner, course, self.method)

            # learned_course = self._sample_mastery_outcome(course[1])      # calculate overlearning

            self._agent_skills = np.maximum(self._agent_skills, course[1])  # learned_course)
            
            # Update preferences covered skills if any
            improved = (self._agent_skills > learner)
            self.covered_want |= improved & (self._want.astype(bool))

            observation = self.get_obs()
            info = self.get_info()
            info["utility"] = utility

            if self.feature == "UIR":
                reward = info["utility"]  # Use utility as reward
            elif self.feature == "EUIR":
                reward = (info["nb_applicable_jobs"]) / len(self.jobs) + info["utility"]  # Combine both metrics (DEPRECATED)
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
            use_standard = self.eval_env.unwrapped.config.get("use_standard", False)   # Standard PPO hyperparameters settings
            if not use_standard:
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
            
            time_start = time.perf_counter()  # Start timing the evaluation
            avg_jobs = 0  # Accumulator for average jobs across learners
            avg_reward = 0

            avg_goal_gap = 0                                     # Accumulator for Nm metric fort evaluation
            avg_pref_cov = 0.0                                   # Accumulator for preference percentage coverage

            avg_total_levels_req = 0                             # Accumulator for total number of LEVELS required
            avg_skills_req_unique = 0                            # Accumulator for total number of SKILLS required
            avg_skills_fully_covered = 0                         # Accumulator for total number of SKILLS fully covered
            avg_skills_missing_unique = 0                        # Accumulator for total number of SKILLS missing

            use_preference = self.eval_env.unwrapped.config.get("use_preference", True)

            # Loop through each learner in the evaluation dataset
            for learner_idx, learner in enumerate(self.eval_env.unwrapped.dataset.learners):
                if use_preference:
                    want, avoid = self.eval_env.unwrapped._eval_want_avoid(learner, learner_idx)
                    self.eval_env.reset(options={"learner": learner, "want": want, "avoid": avoid})  # Reset environment with current learner
                else:
                    self.eval_env.reset(options={"learner": learner})  # Reset environment with current learner

                done = False  # Flag to control evaluation episode
                tmp_avg_jobs = self.eval_env.unwrapped.get_info()["nb_applicable_jobs"]  # Initial jobs applicable without any recommendations
                tmp_avg_reward = 0
                tmp_goal_gap = 0
                tmp_pref_cov = 0.0

                tmp_total_levels_req = 0
                tmp_skills_req_unique = 0
                tmp_skills_fully_covered = 0
                tmp_skills_missing_unique = 0
                

                # Run one full evaluation episode for the learner
                while not done:
                    obs = self.eval_env.unwrapped.get_obs()
                    if isinstance(self.model, MaskablePPO):
                        mask = self.eval_env.get_wrapper_attr("get_action_mask")()
                        action, _state = self.model.predict(obs, action_masks=mask, deterministic=True)
                    else:
                        action, _state = self.model.predict(obs, deterministic=True)  # Predict action using current policy
                    # obs = self.eval_env.get_obs()  # Get current observation (learner's skills)

                    obs, reward, terminated, truncated, info = self.eval_env.step(action)  # Step in environment
                    done = terminated or truncated  # Properly compute done flag

                    # Only update if the recommendation was valid and use nb_applicable_jobs
                    if reward != -1:
                        tmp_avg_jobs = info["nb_applicable_jobs"]

                    tmp_avg_reward = reward
                    
                    if done:
                        tmp_goal_gap = info["goal_gap_total"]
                        tmp_pref_cov = info["pref_coverage"]
                        tmp_total_levels_req = info["total_skill_levels_required"]
                        tmp_skills_req_unique = info["skills_required_unique"]
                        tmp_skills_fully_covered = info["skills_fully_covered"]
                        tmp_skills_missing_unique = info["skills_missing_unique"]

                avg_jobs += tmp_avg_jobs  # Add learner's result to total
                avg_reward += tmp_avg_reward
                avg_goal_gap += tmp_goal_gap
                avg_pref_cov += tmp_pref_cov
                avg_total_levels_req += tmp_total_levels_req
                avg_skills_req_unique += tmp_skills_req_unique
                avg_skills_fully_covered += tmp_skills_fully_covered
                avg_skills_missing_unique += tmp_skills_missing_unique


            time_end = time.perf_counter()  # End timing the evaluation
            
            total_learners = len(self.eval_env.unwrapped.dataset.learners)
            avg = avg_jobs / total_learners
            avg_reward = avg_reward / total_learners
            avg_goal_gap = avg_goal_gap / total_learners
            avg_pref_cov = avg_pref_cov / total_learners
            avg_total_levels_req = avg_total_levels_req / total_learners
            avg_skills_req_unique = avg_skills_req_unique / total_learners
            avg_skills_fully_covered = avg_skills_fully_covered / total_learners
            avg_skills_missing_unique = avg_skills_missing_unique / total_learners



            # Log the result to the console
            print(
                f"Iteration {self.n_calls}. "
                f"Average jobs: {avg} "
                f"Average reward: {avg_reward} "
                f"Average goal gap: {avg_goal_gap} "
                f"Average pref cov: {avg_pref_cov} "
                f"Average total levels req: {avg_total_levels_req} "
                f"Average skills req unique: {avg_skills_req_unique} "
                f"Average skills fully covered: {avg_skills_fully_covered} "
                f"Average skills missing unique: {avg_skills_missing_unique} "
                f"Time: {time_end - time_start:.4f}"
            )

            results_dir = os.path.join(f"{self.eval_env.unwrapped.config['results_path']}{self.eval_env.unwrapped.config['k']}/seed{self.eval_env.unwrapped.config['seed']}")

            # Create directory if it does not already exist
            os.makedirs(results_dir, exist_ok=True)

            file_path = os.path.join(results_dir, self.all_results_filename)

            # Write evaluation result to file
            with open(file_path, self.mode, encoding="utf-8") as f:  # 'w' for first time, 'a' for append afterward
                f.write(
                    f"{self.n_calls} "
                    f"{avg} "
                    f"{avg_reward} "
                    f"{avg_goal_gap} "
                    f"{avg_pref_cov} "
                    f"{avg_total_levels_req} "
                    f"{avg_skills_req_unique} "
                    f"{avg_skills_fully_covered} "
                    f"{avg_skills_missing_unique} "
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
    Numba-compiled helper replicating calculate_course_metrics_gap.
    Returns (Nr, Nm, Nnr) as ints.
    """

    num_jobs, num_skills = jobs.shape

    # After taking the course (max level per skill)
    cons_skills = np.empty_like(learner)
    for s in range(num_skills):
        l = learner[s]
        c = course_provided[s]
        cons_skills[s] = l if l >= c else c

    Nr = 0.0  # total deficit reduction
    Nm = 0.0  # remaining deficits
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
            # Nr: total deficit reduction
            for s in range(num_skills):
                Nr += (missing_before[s] - missing_after[s])
                Nm += missing_after[s]

    # --- Nnr: total gains on irrelevant skills ---
    Nnr = 0.0
    for s in range(num_skills):
        gain = cons_skills[s] - learner[s]
        if gain > 0.0 and not needed[s]:
            Nnr += gain

    return Nr, Nm, Nnr


@njit(cache=True, fastmath=True)
def _calc_metrics_threshold_mastery_numba(
        learner: np.ndarray,
        course_provided: np.ndarray,  # course[1]
        jobs: np.ndarray              # job requirements (l_g), shape (num_jobs, num_skills)
    ) -> tuple:
    """
    Mastery-threshold version:
      missing_before: jobs[j,s] > learner[s]
      covered_after:  max(learner[s], course_provided[s]) >= jobs[j,s]

    Nr: sum over unachievable jobs of (# missing skills that become satisfied after course)
    Nm: sum over unachievable jobs of (# missing skills still not satisfied after course)
    Nnr: # skills increased by course that are not missing for any job before course
    """

    num_jobs, num_skills = jobs.shape

    Nr = 0
    Nm = 0

    # union of missing skills across ALL jobs BEFORE course
    all_missing_any = np.zeros(num_skills, dtype=np.bool_)

    for j in range(num_jobs):
        has_deficit = False
        nr_job = 0
        nm_job = 0

        for s in range(num_skills):
            req = jobs[j, s]
            lv  = learner[s]

            # missing before?
            if req > lv:
                has_deficit = True
                all_missing_any[s] = True

                cv = course_provided[s]
                cons = lv if lv >= cv else cv  # max(lv, cv)

                # mastery threshold for this goal/job:
                if cons >= req:
                    nr_job += 1
                else:
                    nm_job += 1

        if has_deficit:
            Nr += nr_job
            Nm += nm_job

    # Nnr: skills that course increases, and that were NOT missing for any job (before)
    Nnr = 0
    for s in range(num_skills):
        if course_provided[s] > learner[s] and not all_missing_any[s]:
            Nnr += 1

    return Nr, Nm, Nnr