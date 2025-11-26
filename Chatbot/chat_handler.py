# Chatbot/chat_handler.py

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

from UIR.Scripts.Reinforce import Reinforce
from .state import PrefState
from .utils import create_random_profile, filter_jobs_by_skills


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
        searcher: Any,
        dataset: Any,
        device: str = "cpu",
        debug: bool = False,
    ) -> None:
        self.state = state
        self.canon2uid = canon2uid
        self.uid2canon = uid2canon
        self.levels = levels
        self.skills_pool = skills_pool
        self.jobs = jobs
        self.mastery_levels: List[int] = sorted(
            {int(v) for v in levels.values() if int(v) > 0}
        )
        self.searcher = searcher
        self.dataset = dataset
        self.debug = debug

        self.reinforce: Optional[Reinforce] = None
        self.default_k: int = 2

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            "./Chatbot/NER/esco_skill_ner_multi_model/checkpoint-290"
        )
        self.ner_pipeline = pipeline(
            task="token-classification",
            model=self.ner_model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first", # "max",
            device=device
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def handle(self, message: str) -> str:
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
            create_random_profile(
                self.state,
                self.skills_pool,
                self.mastery_levels,
                min_skills=30,
                max_skills=45,
            )
            return (
                "A random user profile has been created. "
                "You can type ':myskills' to view your skills."
            )

        if msg == "clear":
            self.state.clear_preferences()
            return "User profile has been cleared."

        if msg.startswith(":sem "):
            query = msg[5:].strip()
            include_ids, avoid_ids, acquired_ids = self._semantic_ids(query)

            include_pairs = {(self.uid2canon[int(i)], i) for i in include_ids}
            avoid_pairs = {(self.uid2canon[int(i)], i) for i in avoid_ids}
            acquired_pairs = {(self.uid2canon[int(i)], i) for i in acquired_ids}

            self.state.set_include(include_pairs)
            self.state.set_avoid(avoid_pairs)
            self.state.set_acquired(acquired_pairs)

            return f"Semantic include: {len(include_ids)} skills for '{query}'"

        if msg == ":rec":
            learner_vec = self.state.profile.to_skill_vector(self.dataset)
            forbidden = self.forbidden_courses(
                self.state.get_include(),
                self.state.get_avoid(),
            )
            recommendation = self.perform_recommendation(learner_vec, forbidden)
            return f"Here you are: {recommendation}"

        if msg == ":myskills":
            return self._show_skills()

        if msg == ":show":
            return self._show_prefs()

        if msg == ":filter":
            _, msg_out = self._filter_jobs()
            return msg_out

        return "Preferences updated."

    # ------------------------------------------------------------------ #
    # Helpers: inspection                                                 #
    # ------------------------------------------------------------------ #

    def _show_skills(self) -> str:
        if not self.state.profile or not self.state.profile.skills_explicit:
            return "No skills in profile yet. Use 'load resume' first."

        lines = [
            f"- Skill {uid}: level {level}"
            for uid, level in self.state.profile.skills_explicit.items()
        ]
        return "Your current skills:\n" + "\n".join(lines)

    def _show_prefs(self) -> str:
        inc = self.state.get_include()
        avo = self.state.get_avoid()
        return f"[INCLUDE] {inc}\n[AVOID ] {avo}"

    # ------------------------------------------------------------------ #
    # Recommendation + RL                                                #
    # ------------------------------------------------------------------ #

    def perform_recommendation(
        self,
        learner_vec: np.ndarray,
        forbidden_courses: Optional[List[int]] = None,
    ) -> List[int]:
        if self.reinforce is None:
            self._ensure_reinforce()
        return self.reinforce.recommend(learner_vec, forbidden_courses)

    def _ensure_reinforce(self, k: Optional[int] = None) -> None:
        if self.reinforce is not None:
            return

        if k is None:
            k = self.default_k

        self.reinforce = Reinforce(
            dataset=self.dataset,
            model="ppo_mask",
            k=k,
            threshold=self.dataset.config.get("threshold", 0.8),
            run=0,
            save_name=f"chat_k{k}",
            total_steps=0,
            eval_freq=100,
            feature="Weighted-Usefulness-as-Rwd",
            baseline=False,
            method=1,
            beta1=0.1,
            beta2=0.9,
            params=None,
        )

    def forbidden_courses(
        self,
        include_ids: Set[tuple[str, str]],
        avoid_ids: Set[tuple[str, str]],
    ) -> List[int]:
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
        include_ids = {uid for (_canon, uid) in self.state.get_include()}
        avoid_ids = {uid for (_canon, uid) in self.state.get_avoid()}

        filtered = filter_jobs_by_skills(
            self.jobs,
            include_skill_ids=include_ids,
            avoid_skill_ids=avoid_ids,
            level_map=self.levels,
            min_level_num=1,
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
            message: str,
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Extract ESCO skill IDs from a user message via NER + semantic search.
        """
        candidate_spans = self._run_ner_extract_spans(message)

        print(f"Candidate Spans (for Linking): {candidate_spans}")

        include_ids: Set[str] = set()
        avoid_ids: Set[str] = set()
        acquired_ids: Set[str] = set()
        neutral_ids: Set[str] = set()

        for phrase, intent in candidate_spans:
            print(f"phrase: {phrase}")
            matches = self.searcher.search_reranked(phrase)

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

        return include_ids, avoid_ids, acquired_ids

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
