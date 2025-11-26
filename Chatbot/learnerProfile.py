# Chatbot/learnerProfile.py

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional

import numpy as np

Level = int  # -1 = unknown, 1..3


@dataclass
class StudyBudget:
    """Optional study constraints for the learner."""
    k_courses: Optional[int] = None
    time_hours: Optional[int] = None


@dataclass
class Goals:
    """High-level learning or career goals."""
    target_domain: Optional[str] = None
    target_role: Optional[str] = None


@dataclass
class Constraints:
    """Contextual constraints for job/course search."""
    location: Optional[str] = None
    remote: Optional[str] = None    # e.g. "onsite" | "hybrid" | "remote"
    contract: Optional[str] = None
    seniority: Optional[str] = None


@dataclass
class UserProfile:
    """
    Single source of truth about the learner.

    - skills_explicit: skills we know the learner has (from CV, dataset, or chat)
    - skills_inferred: skills inferred by models or heuristics, with confidence
    - domains: user domains or areas of interest
    - goals, study_budget, constraints: collected during the conversation
    """
    skills_explicit: Dict[str, Level] = field(default_factory=dict)                # {skill_uid: level}
    skills_inferred: Dict[str, Tuple[Level, float]] = field(default_factory=dict) # {skill_uid: (level, confidence)}
    domains: Set[str] = field(default_factory=set)

    goals: Goals = field(default_factory=Goals)
    study_budget: StudyBudget = field(default_factory=StudyBudget)
    constraints: Constraints = field(default_factory=Constraints)

    def effective_skills(self, min_conf: float = 0.5) -> Dict[str, Level]:
        """
        Return a merged view of explicit and inferred skills.

        Explicit skills always override inferred ones.
        Only inferred skills with confidence >= min_conf are included.
        """
        out: Dict[str, Level] = dict(self.skills_explicit)
        '''for uid, (level, conf) in self.skills_inferred.items():
            if conf >= min_conf and uid not in out:
                out[uid] = int(level)'''
        return out

    def to_skill_vector(self, dataset, min_conf: float = 0.5) -> np.ndarray:
        """
        Build the learner skill vector aligned with the RL dataset.

        - Length is len(dataset.skills)
        - Uses explicit + inferred skills (above min_conf)
        - Levels <= 0 are ignored
        - dataset.skills2int maps skill_uid (int) -> index in the vector
        """
        nb_skills = len(dataset.skills)
        vec = np.zeros(nb_skills, dtype=np.int32)

        eff = self.effective_skills(min_conf=min_conf)
        for uid, level in eff.items():
            try:
                uid_int = int(uid)
            except (TypeError, ValueError):
                continue

            if int(level) <= 0:
                continue

            if uid_int in dataset.skills2int:
                idx = dataset.skills2int[uid_int]
                vec[idx] = int(level)

        return vec
