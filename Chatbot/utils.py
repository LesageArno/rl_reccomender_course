import random
import re
from typing import List, Dict, Set, Any

import pandas as pd
import numpy as np

from .state import PrefState  # used in type hints


TOKEN_RE = re.compile(r"[a-z0-9.+#\-]+")       # keeps tokens like c++, .net, c#, node.js
CLEAN_RE = re.compile(r"[^a-z0-9\s.+#\-]")     # soft cleaning (preserves . + # -)
_STOP = {
    "i", "you", "he", "she", "we", "they",
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "but",
    "with", "without", "as", "at", "by", "from", "it", "is", "are",
    "be", "been", "being",
}


def normalize(text: str) -> str:
    """Lowercase and remove unwanted characters, keeping a compact whitespace."""
    text = text.lower().strip()
    text = CLEAN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text using a relaxed pattern that preserves technical tokens."""
    return TOKEN_RE.findall(text)


def build_definition_records(
    df: pd.DataFrame,
    canonical_col: str = "Type Level 3",
) -> List[Dict]:
    """
    Build definition records from a taxonomy dataframe.

    Each record has:
      - 'canon': canonical name on the chosen level (normalized)
      - 'uid'  : unique_id from the taxonomy
      - 'def_tokens': set of normalized tokens from the Definition
    """
    if canonical_col not in df.columns:
        canonical_col = "name"

    records: List[Dict] = []
    for _, row in df.iterrows():
        canon_raw = str(row.get(canonical_col, "")).strip()
        uid = row.get("unique_id")
        if not canon_raw or pd.isna(uid):
            continue

        canon = normalize(canon_raw)
        def_norm = normalize(str(row.get("Definition", "")))
        rec = {
            "canon": canon,
            "uid": int(uid),
            "def_tokens": set(tokenize(def_norm)),
        }
        records.append(rec)

    return records


def create_random_profile(
    state: PrefState,
    skills_pool: List[str],
    mastery_levels: List[int],
    min_skills: int = 5,
    max_skills: int = 15,
) -> None:
    """
    Populate state.profile.skills_explicit with a random skill set.
    """
    n = random.randint(min_skills, max_skills)
    picked = random.sample(skills_pool, k=min(n, len(skills_pool)))

    state.profile.skills_explicit.clear()
    for uid in picked:
        level = random.choice(mastery_levels)
        state.profile.skills_explicit[str(uid)] = int(level)


def filter_jobs_by_skills(
    jobs_dict: Dict,
    include_skill_ids: Set[str],
    avoid_skill_ids: Set[str],
    level_map: Dict,
    min_level_num: int = 1,
    include_if_empty: bool = True,
) -> Dict:
    """
    Filter jobs according to include/avoid skill constraints.

    Rules:
      - If a job contains any skill in avoid_skill_ids → discard.
      - If include_skill_ids is not empty:
          keep the job only if it has at least one included skill
          with level >= min_level_num.
      - If include_skill_ids is empty and include_if_empty=True:
          do not apply the include filter.
    """
    norm_level_map = {
        str(k).lower().strip(): int(v) for k, v in level_map.items()
    }

    kept: Dict = {}
    for job_id, pairs in jobs_dict.items():
        # build {skill_id: level_num}
        job_skill_level: Dict[int, int] = {}
        for sid, lvl in pairs:
            lvl_key = str(lvl).lower().strip()
            job_skill_level[int(sid)] = norm_level_map.get(lvl_key, -1)

        # 1) hard avoid
        has_avoid = any(
            int(sid) in job_skill_level for sid in avoid_skill_ids
        )
        if has_avoid:
            continue

        # 2) include
        if include_skill_ids:
            has_include = any(
                (int(sid) in job_skill_level)
                and (job_skill_level[int(sid)] >= min_level_num)
                for sid in include_skill_ids
            )
            if not has_include:
                continue

        kept[job_id] = pairs

    return kept


def filter_jobs_goal_conditioned_tl3(
    jobs_dict: Dict[str, list],
    want: np.ndarray,                 # shape: (n_tl3,), values in {0,1}
    avoid: np.ndarray,                # shape: (n_tl3,), values in {0,1}
    skills2int_tl3: Dict[int, int],   # TL4 uid -> TL3 index
    level_map: Dict[Any, int],
) -> Dict[str, list]:
    """
    Filter TL4 jobs using the same W1-hard goal-job logic as the RL environment (TL3 space).

    A TL3 skill is considered "required" by a job if the job contains at least one TL4 skill
    that maps to that TL3 and has a numeric level > 0.

    Rules (W1-hard):
      - WANT: keep jobs that require at least one wanted TL3 skill (if WANT is empty, do not filter by WANT)
      - AVOID: exclude jobs that require any avoided TL3 skill
      - Fallback: if empty, relax AVOID; if still empty, keep all jobs
    """
    norm_level_map = {str(k).lower().strip(): int(v) for k, v in level_map.items()}

    want_idx = set(np.nonzero(want)[0])
    avoid_idx = set(np.nonzero(avoid)[0])

    job_ids = list(jobs_dict.keys())
    req3_by_job: Dict[str, set[int]] = {}

    for jid, pairs in jobs_dict.items():
        req3 = set()
        for sid, lvl in pairs:
            lvl_num = norm_level_map.get(str(lvl).lower().strip(), -1)
            if lvl_num <= 0:
                lvl_num = 3

            tl3 = skills2int_tl3.get(int(sid))
            if tl3 is not None:
                req3.add(int(tl3))

        req3_by_job[jid] = req3

    # WANT condition
    if not want_idx:
        has_want = {jid: True for jid in job_ids}
    else:
        has_want = {jid: bool(req3_by_job[jid] & want_idx) for jid in job_ids}

    # AVOID condition
    if not avoid_idx:
        has_avoid = {jid: False for jid in job_ids}
    else:
        has_avoid = {jid: bool(req3_by_job[jid] & avoid_idx) for jid in job_ids}

    kept_ids = [jid for jid in job_ids if has_want[jid] and not has_avoid[jid]]

    # Defensive fallback (match RL environment behavior)
    if not kept_ids:
        kept_ids = [jid for jid in job_ids if has_want[jid]]
        if not kept_ids:
            kept_ids = job_ids[:]

    return {jid: jobs_dict[jid] for jid in kept_ids}



def _tok(text: str) -> List[str]:
    """Simple tokenizer: lowercase, strip punctuation, drop stopwords."""
    t = re.sub(r"[^a-z0-9 +#.\-]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return [w for w in t.split() if w and w not in _STOP]


def sliding_ngrams(
    text: str,
    min_n: int = 2,
    max_n: int = 5,
    max_phr_len: int = 40,
) -> List[str]:
    """
    Generate n-gram phrases from text.

    Example:
        "artificial intelligence engineering" →
        ["artificial intelligence",
         "intelligence engineering",
         "artificial intelligence engineering"]
    """
    toks = _tok(text)[:max_phr_len]
    seen: Set[str] = set()
    out: List[str] = []

    for n in range(min_n, max_n + 1):
        for i in range(0, max(0, len(toks) - n + 1)):
            phrase = " ".join(toks[i:i + n])
            if phrase not in seen:
                seen.add(phrase)
                out.append(phrase)

    return out
