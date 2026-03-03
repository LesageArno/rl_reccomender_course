# Chatbot/chat_handler.py

from collections import defaultdict
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pdfplumber
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

from UIR.Scripts.Reinforce import Reinforce
from .state import PrefState
from .utils import create_random_profile, filter_jobs_by_skills, filter_jobs_goal_conditioned_tl3
from .LLMDialogManager import LLMDialogManager
ChatMessage = Dict[str, str]  # {"role": "user" | "assistant" | "system", "content": "..."}


class ChatHandler:
    """Handle user messages and update the preference state."""

    def __init__(
        self,
        state: PrefState,
        canon2uid: Dict[str, int],
        uid2canon: Dict[int, str],
        levels: Dict[str, Any],
        skills_pool: List[str],
        jobs: Dict[str, Any],
        courses_requirements: Dict[str, List],
        courses_acquisitions: Dict[str, List],
        searcher: Any,
        dataset: Any,
        skills2int_tl3: Any,
        n_tl3: Any,
        int2skills_tl3: Any,
        device: str = "cuda",
        debug: bool = False,
    ) -> None:
        self.state = state
        self.canon2uid = canon2uid
        self.uid2canon = uid2canon
        self.levels = levels
        self.skills_pool = skills_pool
        self.jobs = jobs
        self.courses_requirements = courses_requirements
        self.courses_acquisitions = courses_acquisitions
        self.mastery_levels: List[int] = sorted(
            {int(v) for v in levels.values() if int(v) > 0}
        )
        self.searcher = searcher
        self.dataset = dataset
        self.skills2int_tl3 = skills2int_tl3
        self.n_tl3 = n_tl3
        self.int2skills_tl3 = int2skills_tl3
        self.debug = debug

        self.reinforce: Optional[Reinforce] = None
        # self.default_k: int = 2
        self.k_changed: bool = False

        ner_ckpt = "./Chatbot/NER/xml-roberta/checkpoint-2375"

        self.tokenizer = AutoTokenizer.from_pretrained(ner_ckpt)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_ckpt)

        self.ner_pipeline = pipeline(
            task="token-classification",
            model=self.ner_model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first", # "max",
            device=device
        )
        self.llm = LLMDialogManager(
            model_card= "mistralai/Mistral-7B-Instruct-v0.2", # "microsoft/Phi-3-mini-128k-instruct",
            max_new_tokens=120,
            temperature=0.2,
        )
        self.history: List[ChatMessage] = []
        self.max_history_messages = 14

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def handle(self, message: str, cv_text:str = None) -> str:
        """Process a single user message and return a reply string."""
        msg = message.strip().lower()

        if msg in {":quit", "quit"}:
            return "Goodbye!"

        if msg.startswith(":help"):
            return (
                "\nCommands:\n"
                "  ':show'       Show current skill preferences\n"
                "  ':filter'     Count jobs matching current filters\n"
                "  ':rec'        Recommend a course sequence\n"
                "  'load resume' Generate a random learner profile\n"
                "  ':myskills'   Show current explicit skills\n"
                "  'clear'       Clear preferences\n"
                "  ':quit'       Exit the chat\n"
            )

        if "load resume" in msg:
            print("Extracted Resume Text:")
            # print(cv_text)
            reply = self.llm.extract_skills_from_cv_text(
                cv_text=cv_text,
                max_new_tokens=2000
            )
            print(reply)

            self._create_profile_from_CV(reply)

            '''create_random_profile(
                self.state,
                self.skills_pool,
                self.mastery_levels,
                min_skills=30,
                max_skills=45,
            )'''
            self.k_changed = True
            return (
                #"A random user profile has been created. "
                "Your resume has been processed and your profile updated. "
                "You can press 'my skills' button to view your skills."
            )

        if msg == "clear":
            self.state.clear_preferences()
            return "User profile has been cleared."

        if msg.startswith(":sem "):
            query = msg[5:].strip()

            # 🧠 1) GATE LLM
            has_skills = self.llm.detect_skill_presence(query)

            if not has_skills:
                # ❌ no skill → only LLM ANSWER
                reply = self.llm.chat(
                    user_input=query,
                    history=self.history,
                    system_prompt=None,
                    extra_context=None,
                    max_new_tokens=500, #120,
                    temperature=0.2,
                )

                self.history.append({"role": "user", "content": message})
                self.history.append({"role": "assistant", "content": reply})
                self.history[:] = self.history[-self.max_history_messages:]

                return reply + "\n" + "Remember that you can always modify your profile by using the interface on the right pane."

            try:
                skills = self.llm.extract_structured_preferences(query, history=self.history)
                # Convert LLM JSON -> spans like NER
                candidate_spans = []
                candidate_spans += [(x["text"], "INCLUDE_SKILL") for x in skills.get("include", [])]
                candidate_spans += [(x["text"], "AVOID_SKILL") for x in skills.get("avoid", [])]
                candidate_spans += [(x["text"], "ACQUIRED_SKILL") for x in skills.get("acquired", [])]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                candidate_spans = self._run_ner_extract_spans(query)

            print(candidate_spans)

            include_ids, avoid_ids, acquired_ids, _ = self._semantic_ids(candidate_spans)

            levels_by_uid = self.llm.infer_mastery_levels(
                user_text=query,
                acquired_uids=acquired_ids,
                uid2canon=self.uid2canon,
            )

            include_pairs = {(self.uid2canon[int(i)], i) for i in include_ids}
            avoid_pairs = {(self.uid2canon[int(i)], i) for i in avoid_ids}
            acquired_pairs = {(self.uid2canon[int(i)], i, levels_by_uid.get(i, 1)) for i in acquired_ids}

            self.state.set_include(include_pairs)
            self.state.set_avoid(avoid_pairs)
            self.state.set_acquired(acquired_pairs)
            
            reply = self.llm.explain_updated_preferences(
                original_text=query,
                include_pairs=include_pairs,
                avoid_pairs=avoid_pairs,
                acquired_pairs=acquired_pairs,
                #history=self.history
            )

            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": reply})

            # Only last N messages
            self.history[:] = self.history[-self.max_history_messages:]

            return reply 

        if msg == ":rec":
            include_pairs = self.state.get_include()
            avoid_pairs = self.state.get_avoid()

            want_vec, avoid_vec = self.build_tl3_preference_vectors(include_pairs, avoid_pairs)

            learner_vec = self.state.profile.to_tl3_skill_vector(
                skills2int_tl3=self.skills2int_tl3,
                n_tl3=self.n_tl3,
            )
            #learner_vec = self.state.profile.to_skill_vector(self.dataset)

            #forbidden = self.forbidden_courses(
            #    self.state.get_include(),
            #    self.state.get_avoid(),
            #)
            recommendation = self.perform_recommendation(learner_vec,want_vec, avoid_vec)#, forbidden)
            
            jobs_goal = recommendation["jobs_goal"]
            
            #print(recommendation['nb_applicable_jobs'])
            courses = {}                                              # In this there will be all the skill associated to course ids
            skills_learned = {}
            for course_id in recommendation["seq_course_codes"]:
                print(course_id)
                courses[course_id] = []
                names = []
                for skill in self.courses_acquisitions.get(course_id, []):
                    print(f"Skills uid {skill}, name {self.uid2canon.get(skill[0], 'Unknown Skill')}")
                    skill_name = self.uid2canon.get(skill[0], "Unknown Skill")
                    if skill_name in names:
                        continue
                    names.append(skill_name)
                    skill_level = skill[1]
                    skills_learned[skill_name] = (skill[1], skill[0])           ###### SKILL[0] JUST TO SEE THE UID ASSOCIATED TO IT
                    courses[course_id].append((skill_name, skill_level))

            learned_vec = self.skills_learned_to_tl3_vec(skills_learned, self.skills2int_tl3, self.n_tl3)
            debug = self.debug_job_skills(jobs_goal, want_vec, avoid_vec, learner_vec, learned_vec)
            debug["nb_applicable_jobs"] = recommendation["nb_applicable_jobs"]
            debug["recommended_courses"] = courses
            print(courses)

            for metric in debug:
                print(f"{metric}: {debug[metric]}")

            if skills_learned == {}:
                inc = self.state.get_include()
                avo = self.state.get_avoid()

                return (
                    "I couldn't recommend any courses with your current preferences. "
                    f"You asked to include: {inc}, and avoid: {avo}. "
                    "With these constraints, every course is excluded (any course that matches your include set also contains at least one avoided skill). "
                    "Would you like to adjust your preferences (remove/relax an avoid skill, or add/relax an include skill) so I can generate recommendations?"
                )

            reply = self.llm.build_recommendation_context(
                history=None, #self.history,
                course_ids=recommendation["seq_course_codes"],
                skills_learned=skills_learned,
                include_pairs=self.state.get_include(),
                avoid_pairs=self.state.get_avoid(),
                acquired_pairs=self.state.get_acquired(),
                max_new_tokens=500
            )

            rep = f" \
            Here are your recommended courses: {recommendation['seq_course_codes']} \n \
            Skills you will learn: {skills_learned} \
            "
            rep = f"Here are your recommended courses: {recommendation['seq_course_codes']} \n" #\
                    #Check insights on the window on the right to have further details about this recommendation. \n"
            
            insight = f"Please have a look at the insights on the right pane so you have a better understanding of \
                    the recommendation, and if you are not fine with it try to adjust your profile accordingly and try again."
            return rep + "\n" + reply + "\n" + insight, debug
            #return rep + "\n" + "\n" + reply, debug #f"Here you are: {recommendation}"

        if msg == ":myskills":
            reply = self._show_skills()

            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": reply})

            # Only last N messages
            self.history[:] = self.history[-self.max_history_messages:]

            return reply
            
        if msg == ":show":
            reply = self._show_prefs()

            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": reply})

            # Only last N messages
            self.history[:] = self.history[-self.max_history_messages:]
            return reply

        if msg == ":filter":
            filtered_jobs, msg_out = self._filter_jobs()
            jobs_list = "\n".join(map(str, filtered_jobs.keys()))
            return f"```\n{jobs_list}\n```\n{msg_out}"

        reply = self.llm.chat(
            user_input=message,       
            history=None,             
            system_prompt=None,
            extra_context=None,      
            max_new_tokens=120,
            temperature=0.2,
        )    

        return reply

    # ------------------------------------------------------------------ #
    # Helpers: inspection                                                 #
    # ------------------------------------------------------------------ #
    def debug_job_skills(
        self,
        jobs_goal,        # shape (J, S)
        want_uids,        # shape (S,)
        avoid_uids,       # shape (S,)
        acquired_uids,    # shape (S,)
        skills_learned,   # shape (S,)
    ):
        """
        Level-aware debug over multiple job goals (TL3-aligned).
        Counts are computed on UNIQUE TL3 indices across all jobs (no double counting per job).
        """

        jobs_goal = np.asarray(jobs_goal)
        want_uids = np.asarray(want_uids)
        avoid_uids = np.asarray(avoid_uids)
        acquired_uids = np.asarray(acquired_uids)
        skills_learned = np.asarray(skills_learned)

        # after levels (0..3)
        after = np.maximum(acquired_uids, skills_learned).clip(0, 3)

        # --- skill uniche (presenza, ignora livelli) ---
        jobs_presence = (jobs_goal > 0).any(axis=0)          # (46,) bool
        learned_presence = (skills_learned > 0)              # (46,) bool
        want_presence = (want_uids > 0)
        avoid_presence = (avoid_uids > 0)

        want_in_jobs_goal = int(np.sum(want_presence & jobs_presence))
        want_in_learned = int(np.sum(want_presence & learned_presence))
        avoid_in_learned = int(np.sum(avoid_presence & learned_presence))

        # --- livelli coperti (max 3 per skill) ---
        # richiesto: somma livelli su job, ma ogni skill max 3
        required_levels = jobs_goal.sum(axis=0).clip(0, 3)   # (46,)
        covered_levels = np.minimum(required_levels, after)  # (46,)

        levels_required_total = int(required_levels.sum())
        levels_covered_total = int(covered_levels.sum())
        levels_missing_total = int((required_levels - covered_levels).sum())

        required_presence = required_levels > 0
        fully_covered_presence = required_presence & (after >= required_levels)

        skills_required_unique = int(required_presence.sum())
        skills_fully_covered_unique = int(fully_covered_presence.sum())
        skills_not_fully_covered_unique = skills_required_unique - skills_fully_covered_unique

        # livelli mancanti per skill
        levels_missing = (required_levels - covered_levels).clip(0)

        # quante volte la skill compare nei job (presenza, non livelli)
        job_presence_count = (jobs_goal > 0).sum(axis=0)

        # prendo solo skill davvero mancanti
        missing_mask = levels_missing > 0

        # ranking: prima quelle più presenti nei job, poi più livelli mancanti
        ranking = sorted(
            [
                {
                    "skill_idx": int(i),
                    "skill_name": self.int2skills_tl3[int(i)],
                    
                    "levels_required": int(required_levels[i]),
                    "levels_missing": int(levels_missing[i]),
                    "job_coverage": int(job_presence_count[i]),
                }
                for i in np.where(required_levels)[0] #missing_mask)[0]
            ],
            key=lambda x: (-x["job_coverage"], -x["levels_missing"])
        )

        print("####################################################")
        #print(ranking)
        self.print_skill_ranking(ranking, 46)
        print("####################################################")

        extra_learned_mask = learned_presence & (~jobs_presence)
        extra_ranking = sorted(
            [
                {
                    "skill_idx": int(i),
                    "skill_name": self.int2skills_tl3[int(i)],
                    "learned_level": int(skills_learned[i]),
                    "job_coverage": int((jobs_goal > 0).sum(axis=0)[i]),  # sarà 0 per definizione
                }
                for i in np.where(extra_learned_mask)[0]
            ],
            key=lambda x: (-x["learned_level"], x["skill_name"])  # prima livello, poi nome
        )
        print("####################################################")
        self.print_extra_skill_ranking(extra_ranking, 46)
        print("####################################################")


        return {
            "jobs_count": int(jobs_goal.shape[0]),

            "total_want": sum(want_uids),
            "want_in_jobs_goal_unique": want_in_jobs_goal,
            "want_in_learned_unique": want_in_learned,

            "total_avoid": sum(avoid_uids),
            "avoid_in_learned_unique": avoid_in_learned,

            "levels_required_total": levels_required_total,
            "levels_covered_total": levels_covered_total,
            "levels_missing_total": levels_missing_total,

            "skills_required_unique": skills_required_unique,
            "skills_fully_covered_unique": skills_fully_covered_unique,
            "skills_not_fully_covered_unique": skills_not_fully_covered_unique,

            "ranking": ranking,
            "extra_ranking": extra_ranking,
        }
    
    def print_skill_ranking(self, ranking, top_k=10):
        print("\n=== Skill mancanti (priorità) ===")
        for r in ranking[:top_k]:
            print(
                f"- {r['skill_name']} | "
                f"jobs:{r['job_coverage']} | "
                f"req_lv:{r['levels_required']} | "
                f"missing_lv:{r['levels_missing']}"
            )

    def print_extra_skill_ranking(self, ranking, top_k=10):
        print("\n=== Skill mancanti (priorità) ===")
        for r in ranking[:top_k]:
            print(
                f"- {r['skill_name']} | "
                f"jobs:{r['job_coverage']} | "
                f"learned_lv:{r['learned_level']}"
            )






    def _show_skills(self) -> str:
        '''
        Show current possessed skills in the profile.
        '''
        if not self.state.profile or not self.state.profile.skills_explicit:
            return "No skills in profile yet. Use 'load resume' first."
        
        acq_skills = self.state.get_acquired()

        lines = [
            f"- Skill {uid}: level {level}, name '{name}'"
            for name, uid, level in acq_skills
        ]
        return "Your current skills:\n" + "\n".join(lines)

    def _show_prefs(self) -> str:
        '''
        Show current skill preferences.
        '''
        inc = self.state.get_include()
        avo = self.state.get_avoid()
        return f"[INCLUDE] {inc}\n[AVOID ] {avo}"

    # ------------------------------------------------------------------ #
    # Recommendation + RL                                                #
    # ------------------------------------------------------------------ #

    def perform_recommendation(
        self,
        learner_vec: np.ndarray,
        want_vec: np.ndarray,
        avoid_vec: np.ndarray,
        forbidden_courses: Optional[List[int]] = None,
    ) -> List[int]:
        '''
        Perform course recommendation using RL.
        
        :param learner_vec: Vector representing the learner's skills
        :param forbidden_courses: List of course indices to avoid
    
        :return: list of recommended course indices
        '''
        if self.k_changed:
            self._ensure_reinforce()
        return self.reinforce.recommend(learner_vec,want=want_vec, avoid=avoid_vec, forbidden=None)

    def _ensure_reinforce(self) -> None:
        '''
        Ensure that the Reinforce instance is created and up-to-date.
        '''
        k = self.state.get_k()
        print(k)
        
        self.reinforce = Reinforce(
            dataset=self.dataset,
            config=self.dataset.config,
            k=k,
            use_pretrained=True
        )
        self.k_changed = False  # reset change flag

    def forbidden_courses(
        self,
        include_ids: Set[tuple[str, str]],
        avoid_ids: Set[tuple[str, str]],
    ) -> List[int]:
        '''
        Determine forbidden courses based on include and avoid skill IDs.

        :param include_ids: set of (name, uid) pairs to include
        :param avoid_ids: set of (name, uid) pairs to avoid

        :return: list of forbidden course indices
        '''
        inc = {
            self.dataset.skills2int[int(s)]
            for _, s in include_ids
            if int(s) in self.dataset.skills2int
        }
        avo = {
            self.dataset.skills2int[int(s)]
            for _, s in avoid_ids
            if int(s) in self.dataset.skills2int
        }

        forbidden: List[int] = []
        for i, course in enumerate(self.dataset.courses):
            provided = set(np.nonzero(course[1] > 0)[0])

            if provided & avo:
                forbidden.append(i)
            elif inc and not (provided & inc):
                forbidden.append(i)

        return forbidden

    # ------------------------------------------------------------------ #
    # Job filtering                                                      #
    # ------------------------------------------------------------------ #

    def _filter_jobs(self) -> Tuple[Dict[str, Any], str]:
        '''
        Filter jobs based on current skill preferences.
        
        :return: (filtered_jobs, message)
        '''
        '''include_ids = {uid for (_canon, uid) in self.state.get_include()}
        avoid_ids = {uid for (_canon, uid) in self.state.get_avoid()}

        filtered = filter_jobs_by_skills(
            self.jobs,
            include_skill_ids=include_ids,
            avoid_skill_ids=avoid_ids,
            level_map=self.levels,
            min_level_num=1,
        )'''
        include_pairs = self.state.get_include()  # Set[Tuple[canon, uid_str]]
        avoid_pairs = self.state.get_avoid()      # Set[Tuple[canon, uid_str]]

        want, avoid = self.build_tl3_preference_vectors(
            include_pairs=include_pairs,
            avoid_pairs=avoid_pairs,
            conflict_policy="include_wins",
        )

        filtered = filter_jobs_goal_conditioned_tl3(
            jobs_dict=self.jobs,
            want=want,
            avoid=avoid,
            skills2int_tl3=self.skills2int_tl3,
            level_map=self.levels,
        )
        return filtered, f"{len(filtered)} jobs match your current filters."

    # ------------------------------------------------------------------ #
    # NER + semantic ESCO matching                                      #
    # ------------------------------------------------------------------ #

    def _run_ner_extract_spans(self, message: str) -> List[Tuple[str, str]]:
        """
        Run the NER pipeline and return (span_text, polarity_label) pairs.
        """
        try:
            ner_results = self.ner_pipeline(message)
        except Exception as e:  # noqa: BLE001
            print(f"NER Pipeline Error: {e}")
            return []

        spans: List[Tuple[str, str]] = []
        for result in ner_results:
            entity_group = result.get("entity_group", "")
            if entity_group.endswith("_SKILL"):
                span_text = result["word"]
                spans.append((span_text, entity_group))
                print(f"DEBUG: Extracted Span: '{span_text}' -> {entity_group}")
        return spans

    def _semantic_ids(
            self,
            candidate_spans: List[Tuple[str, str]],
    ) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        """
        Extract ESCO skill IDs from a user message via NER + semantic search.
        """
        print(f"Candidate Spans (for Linking): {candidate_spans}")

        include_ids: Set[str] = set()
        avoid_ids: Set[str] = set()
        acquired_ids: Set[str] = set()
        neutral_ids: Set[str] = set()

        for phrase, intent in candidate_spans:
            print(f"phrase: {phrase}")
            matches = self.searcher.search_reranked(phrase, min_ce=0.6)

            for uid_int, similarity in matches:
                uid = str(uid_int)
                if intent.startswith("INCLUDE"):
                    include_ids.add(uid)
                elif intent.startswith("AVOID"):
                    avoid_ids.add(uid)
                elif intent.startswith("ACQUIRED"):
                    acquired_ids.add(uid)
                else:
                    neutral_ids.add(uid)

        print(f"Final ESCO INCLUDE UIDs (Include): {include_ids}")
        print(f"Final ESCO AVOIDs: {avoid_ids}")
        print(f"Final ESCO AQUIRED UIDs: {acquired_ids}")

        return include_ids, avoid_ids, acquired_ids, neutral_ids
    

    def _create_profile_from_CV(self, cv_json: List[Dict[str, Any]]) -> None:
        """
        Create or update the user profile based on extracted CV skills.
        """
        possessed_ids: Set[Tuple[str, int]] = set()
        for skill_entry in cv_json:
            skill_name = skill_entry.get("skill_name", "")
            skill_level = skill_entry.get("level", 1)

            matches = self.searcher.search_reranked(skill_name, min_ce=0.6)
            for uid_int, _ in matches:
                uid = str(uid_int)
                possessed_ids.add((uid, skill_level))
        
        possessed = {(self.uid2canon[int(id)], id, skill_level) for id, skill_level in possessed_ids}

        self.state.set_acquired(possessed)
        print(f"Profile updated with {len(possessed)} acquired skills from CV.")


    # ------------------------------------------------------------------ #
    # Legacy / experimental semantic method                              #
    # ------------------------------------------------------------------ #

    def _semantic_ids3(self, message: str):
        """
        Extract semantically relevant skill IDs from a long user message.
        Implementation notes:
        - Use dynamic n-grams to generate compact candidate phrases.
        - Query the semantic index per phrase and collect evidence per skill.
        - Keep external behavior unchanged: return a set of TL4 ids.
        """
        include_phrases, avoid_phrases, other_phrases = self.intent_detector.windows_to_polar_phrases(message)

        print(f"Include phrases: {include_phrases}")
        print(f"Avoid phrases: {avoid_phrases}")
        print(f"Other phrases: {other_phrases}")

        if not include_phrases and not avoid_phrases:
            return (), ()

        scores = {}

        for phrase in include_phrases:
            matches = self.searcher.search_reranked(phrase)
            for uid_int, sim in matches:
                uid = str(uid_int)
                b = scores.get(uid)
                if b is None:
                    scores[uid] = {
                        "hits": 1,
                        "sum_sim": float(sim),
                        "best_sim": float(sim),
                        "best_phrase": phrase,
                    }
                else:
                    b["hits"] += 1
                    b["sum_sim"] += float(sim)
                    if sim > b["best_sim"]:
                        b["best_sim"] = float(sim)
                        b["best_phrase"] = phrase

        include_ids = set(scores.keys())

        avoid_ids = set()
        for phr in avoid_phrases:
            for uid_int, _ in self.searcher.search_reranked(phr):
                avoid_ids.add(str(uid_int))

        return include_ids - avoid_ids, avoid_ids



    # ------------------------------------------------------------------ #
    # TL3 mapping for compatibility                                      #
    # ------------------------------------------------------------------ #
    def build_tl3_preference_vectors(
        self,
        include_pairs: Set[Tuple[str, str]],
        avoid_pairs: Set[Tuple[str, str]],
        *,
        conflict_policy: str = "include_wins",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert TL4 include/avoid (uid as str) -> TL3 one-hot vectors (len = self.n_tl3).
        Uses mapping built to match RL training: self.skills2int_tl3[tl4_uid] = tl3_index.
        """
        want = np.zeros(self.n_tl3, dtype=np.float32)
        avoid = np.zeros(self.n_tl3, dtype=np.float32)

        def _uid_to_tl3_idx(uid: str):
            try:
                return self.skills2int_tl3.get(int(uid))
            except Exception:
                return None

        for _name, uid in include_pairs:
            idx = _uid_to_tl3_idx(uid)
            if idx is not None:
                want[idx] = 1.0

        for _name, uid in avoid_pairs:
            idx = _uid_to_tl3_idx(uid)
            if idx is not None:
                avoid[idx] = 1.0

        # resolve conflicts
        conflict = (want > 0) & (avoid > 0)
        if np.any(conflict):
            if conflict_policy == "include_wins":
                avoid[conflict] = 0.0
            elif conflict_policy == "avoid_wins":
                want[conflict] = 0.0
            elif conflict_policy == "zero_both":
                want[conflict] = 0.0
                avoid[conflict] = 0.0

        return want, avoid
    

    def skills_learned_to_tl3_vec(self, skills_learned, skills2int_tl3, n_tl3, replace_unk=2):
        """
        Convert skills_learned {name: (level_str, tl4_uid)} -> TL3 vector with mastery levels (0..3),
        using the same logic as Dataset.get_avg_skills(): map strings -> int, replace unknown (-1),
        average duplicates, round.
        """
        buckets = defaultdict(list)  # tl3_idx -> [levels...]

        for (lvl_str, tl4_uid) in skills_learned.values():

            if not (isinstance(lvl_str, str) and lvl_str in self.levels):
                continue

            lvl = self.levels[lvl_str]
            if lvl == -1:
                lvl = replace_unk

            idx = skills2int_tl3.get(int(tl4_uid))
            if idx is not None:
                buckets[idx].append(int(lvl))

        vec = np.zeros(n_tl3, dtype=int)
        for idx, lvls in buckets.items():
            vec[idx] = int(round(sum(lvls) / len(lvls)))

        print(vec)

        return vec
