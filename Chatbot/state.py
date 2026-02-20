# Chatbot/state.py

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Literal, Any

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
    #skills: Dict[SkillUID, SkillEntry] = field(default_factory=dict)
    skills: Dict[SkillUID, Dict[str, Any]] = field(default_factory=dict)

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

    def get_include(self, unique: bool = True) -> Set[Tuple[str, SkillUID]]:
        """
        Return all skills marked as 'include'.

        :param unique: If True, collapse multiple UIDs that share the same
                    canonical name (e.g., TL4 under same TL3).
        :return: Set of (skill_name, skill_uid)
        """
        include_skills = []

        for skill_uid, skill_data in self.skills.items():
            if skill_data.get("include", False):
                include_skills.append((skill_data["name"], skill_uid))

        if not unique:
            return set(include_skills)

        seen_names = set()
        filtered_skills = set()

        for skill_name, skill_uid in include_skills:
            if skill_name not in seen_names:
                seen_names.add(skill_name)
                filtered_skills.add((skill_name, skill_uid))

        return filtered_skills

    def get_avoid(self, unique: bool = True) -> Set[Tuple[str, SkillUID]]:
        """
        Return all skills marked as 'avoid'.

        :param unique: If True, collapse multiple UIDs sharing the same name.
        :return: Set of (skill_name, skill_uid)
        """
        avoid_skills = []

        for skill_uid, skill_data in self.skills.items():
            if skill_data.get("avoid", False):
                avoid_skills.append((skill_data["name"], skill_uid))

        if not unique:
            return set(avoid_skills)

        seen_names = set()
        filtered_skills = set()

        for skill_name, skill_uid in avoid_skills:
            if skill_name not in seen_names:
                seen_names.add(skill_name)
                filtered_skills.add((skill_name, skill_uid))

        return filtered_skills

    '''def get_acquired(self) -> Set[Tuple[str, SkillUID, SkillLevel]]:
        """Return skills marked as 'acquired' as (name, uid)."""
        acquired = set()
        for uid, (name, pol) in self.skills.items():
            if pol == "acquired":
                skill_level = 1
                if self.profile and uid in self.profile.skills_explicit:
                    skill_level = self.profile.skills_explicit[uid]
                acquired.add((name, uid, skill_level))
        return acquired'''
    def get_acquired(self, unique: bool = True) -> Set[Tuple[str, SkillUID, SkillLevel]]:
        """
        Return all skills marked as 'acquired'.

        The proficiency level is retrieved from the learner profile.
        If not explicitly stored, a default level of 1 is assumed.

        :param unique: If True, collapse multiple UIDs sharing the same name.
        :return: Set of (skill_name, skill_uid, skill_level)
        """
        acquired_skills = []

        for skill_uid, skill_data in self.skills.items():
            if skill_data.get("acquired", False):
                skill_level = self.profile.skills_explicit.get(skill_uid, 1)
                acquired_skills.append(
                    (skill_data["name"], skill_uid, skill_level)
                )

        if not unique:
            return set(acquired_skills)

        seen_names = set()
        filtered_skills = set()

        for skill_name, skill_uid, skill_level in acquired_skills:
            if skill_name not in seen_names:
                seen_names.add(skill_name)
                filtered_skills.add((skill_name, skill_uid, skill_level))

        return filtered_skills
    
    def set_include(self, entries: Set[Tuple[str, SkillUID]]) -> None:
        """Mark the given skills as 'include'."""
        for name, uid in entries:

            if uid not in self.skills:
                self.skills[uid] = {
                    "name": name,
                    "acquired": False,
                    "include": False,
                    "avoid": False
                }

            self.skills[uid]["include"] = True
            self.skills[uid]["avoid"] = False

    def set_avoid(self, entries: Set[Tuple[str, SkillUID]]) -> None:
        """Mark the given skills as 'avoid'."""
        for name, uid in entries:

            if uid not in self.skills:
                self.skills[uid] = {
                    "name": name,
                    "acquired": False,
                    "include": False,
                    "avoid": False
                }

            self.skills[uid]["avoid"] = True
            self.skills[uid]["include"] = False

    def set_acquired(self, entries: Set[Tuple[str, SkillUID, SkillLevel]]) -> None:
        """Mark the given skills as 'acquired'."""
        for name, uid, level in entries:

            if uid not in self.skills:
                self.skills[uid] = {
                    "name": name,
                    "acquired": False,
                    "include": False,
                    "avoid": False
                }

            self.skills[uid]["acquired"] = True

            if self.profile:
                current = self.profile.skills_explicit.get(uid, 0)
                if level > current:
                    self.profile.skills_explicit[uid] = level

    def remove_by_uids(self, skill_uids: Set[SkillUID], polarity: SkillPolarity) -> int:
        """
        Remove a specific polarity flag from the given skill UIDs.

        Only the requested polarity is removed.
        Other flags (acquired/include/avoid) remain untouched.

        If after removal the skill has no active flags,
        the skill entry is fully deleted from state.

        :param skill_uids: Set of skill UIDs to modify
        :param polarity: One of "acquired", "include", "avoid"
        :return: Number of updated skills
        """
        removed_count = 0

        for skill_uid in skill_uids:
            if skill_uid not in self.skills:
                continue

            skill_data = self.skills[skill_uid]

            if not skill_data.get(polarity, False):
                continue

            # Remove only the specified polarity
            skill_data[polarity] = False
            removed_count += 1

            # If removing acquired, also remove from learner profile
            if polarity == "acquired" and skill_uid in self.profile.skills_explicit:
                del self.profile.skills_explicit[skill_uid]

            # If no flags remain, delete the skill entirely
            if not (
                skill_data["acquired"]
                or skill_data["include"]
                or skill_data["avoid"]
            ):
                del self.skills[skill_uid]

        return removed_count
    
    def remove_by_names(self, skill_names: Set[str], polarity: SkillPolarity) -> int:
        """
        Remove a specific polarity from all skills matching the given names.

        :param skill_names: Canonical skill names to remove
        :param polarity: One of "acquired", "include", "avoid"
        :return: Number of updated skills
        """
        matching_uids = {
            skill_uid
            for skill_uid, skill_data in self.skills.items()
            if skill_data["name"] in skill_names
            and skill_data.get(polarity, False)
        }

        return self.remove_by_uids(matching_uids, polarity)

    def clear_preferences(self) -> None:
        """Remove all preference labels (include/avoid/acquired) and target roles."""
        self.skills.clear()
        self.target_roles.clear()
        self.profile.skills_explicit.clear()
