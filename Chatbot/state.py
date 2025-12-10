# Chatbot/state.py

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Literal

from .learnerProfile import UserProfile

SkillPolarity = Literal["include", "avoid", "acquired"]
SkillUID = str                         # skill unique id as string
SkillLevel = int                       # skill proficiency level
SkillEntry = Tuple[str, SkillPolarity] # (canonical_name, polarity)


@dataclass
class PrefState:
    """
    Conversation-level preference state.

    - skills: mapping from skill UID -> (canonical name, polarity)
    - target_roles: optional set of target roles mentioned by the user
    - profile: underlying user profile (skills, goals, constraints, ...)
    """
    skills: Dict[SkillUID, SkillEntry] = field(default_factory=dict)
    target_roles: Set[str] = field(default_factory=set)
    profile: UserProfile = field(default_factory=UserProfile)

    def set_k(self, k: int) -> None:
        """Set the number of courses to recommend in the study budget."""
        self.profile.study_budget.k_courses = k

    def get_k(self) -> int:
        """Return the number of courses to recommend, if set in the study budget."""
        if self.profile and self.profile.study_budget.k_courses is not None:
            return self.profile.study_budget.k_courses
        return 2  # default value

    def get_include(self) -> Set[Tuple[str, SkillUID]]:
        """Return skills marked as 'include' as (name, uid)."""
        return {
            (name, uid)
            for uid, (name, pol) in self.skills.items()
            if pol == "include"
        }

    def get_avoid(self) -> Set[Tuple[str, SkillUID]]:
        """Return skills marked as 'avoid' as (name, uid)."""
        return {
            (name, uid)
            for uid, (name, pol) in self.skills.items()
            if pol == "avoid"
        }

    def get_acquired(self) -> Set[Tuple[str, SkillUID, SkillLevel]]:
        """Return skills marked as 'acquired' as (name, uid)."""
        acquired = set()
        for uid, (name, pol) in self.skills.items():
            if pol == "acquired":
                skill_level = 1
                if self.profile and uid in self.profile.skills_explicit:
                    skill_level = self.profile.skills_explicit[uid]
                acquired.add((name, uid, skill_level))
        return acquired
    
    def set_include(self, entries: Set[Tuple[str, SkillUID]]) -> None:
        """Mark the given skills as 'include'."""
        for name, uid in entries:
            self.skills[uid] = (name, "include")

    def set_avoid(self, entries: Set[Tuple[str, SkillUID]]) -> None:
        """Mark the given skills as 'avoid'."""
        for name, uid in entries:
            self.skills[uid] = (name, "avoid")

    def set_acquired(self, entries: Set[Tuple[str, SkillUID, SkillLevel]], default_level: int = 1) -> None:
        """Mark the given skills as 'acquired'."""
        for name, uid, skill_level in entries:
            self.skills[uid] = (name, "acquired")

            # learner profile: add explicit skill if missing
            if self.profile is not None:
                if uid not in self.profile.skills_explicit:
                    self.profile.skills_explicit[uid] = skill_level
                else:
                    # update level if higher
                    current_level = self.profile.skills_explicit[uid]
                    if skill_level > current_level:
                        self.profile.skills_explicit[uid] = skill_level

    def clear_preferences(self) -> None:
        """Remove all preference labels (include/avoid/acquired) and target roles."""
        self.skills.clear()
        self.target_roles.clear()
