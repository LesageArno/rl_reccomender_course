from collections import defaultdict
import json
import pandas as pd
import random
from typing import Dict, Any, Tuple

from .utils import normalize
from .taxonomy_index import build_alias_index

# Paths for chatbot runtime
TAXONOMY_CSV = "Data - Collection/Final/taxonomy.csv"
JOBS_JSON = "Data - Collection/Final/jobs.json"
LEVELS_JSON = "Data - Collection/Final/mastery_levels.json"
COURSES_JSON = "Data - Collection/Final/courses.json"


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return a Python dict."""
    with open(path, "r") as f:
        return json.load(f)
    
def subsample_jobs_like_dataset(jobs: dict, nb_jobs: int = 100, seed: int = 42) -> dict:
    """
    Replica Dataset.get_subsample() per i jobs:
    - ordine base = ordine delle chiavi nel JSON
    - sample deterministico con random.seed(seed)
    - ordine finale = ordine dei job selezionati (come random.sample)
    """
    if nb_jobs >= len(jobs) or nb_jobs <= 0:
        return jobs
    job_ids_in_order = list(jobs.keys())  # stesso ordine di jobs.items() nel Dataset
    random.seed(seed)
    picked_indices = random.sample(range(len(job_ids_in_order)), nb_jobs)
    picked_job_ids = [job_ids_in_order[i] for i in picked_indices]
    return {job_id: jobs[job_id] for job_id in picked_job_ids}


def canon_to_unique_id_map(df: pd.DataFrame, canonical_col: str = "Type Level 4") -> Dict[str, int]:
    """
    Build a mapping canonical_name → unique_id using the taxonomy file.

    Normalization is applied on the canonical label, and unique_id is cast to int.
    """
    if canonical_col not in df.columns:
        canonical_col = "name"

    df["canon"] = df[canonical_col].map(normalize)

    return {
        row["canon"]: int(row["unique_id"])
        for _, row in df.iterrows()
        if pd.notnull(row["canon"]) and pd.notnull(row["unique_id"])
    }

def canon_to_uid_maps(df: pd.DataFrame, canonical_col: str):
    df = df.copy()
    df["canon"] = df[canonical_col].map(normalize)

    canon2uids = defaultdict(list)
    for _, row in df.iterrows():
        canon = row["canon"]
        uid = row["unique_id"]
        if pd.notnull(canon) and pd.notnull(uid):
            canon2uids[canon].append(int(uid))

    # se ogni canon ha 1 uid -> int, altrimenti lista
    canon2uid = {c: (uids[0] if len(uids) == 1 else uids) for c, uids in canon2uids.items()}

    uid2canon = {}
    for c, uids in canon2uids.items():
        for uid in uids:
            uid2canon[uid] = c

    return canon2uid, uid2canon

def build_skills2int_tl3_from_taxonomy(df_taxonomy: pd.DataFrame) -> Tuple[Dict[int, int], int]:
    """
    Build TL4 unique_id -> TL3 index mapping exactly like Dataset.load_skills(level_3=True).

    Returns:
      skills2int_tl3: dict[int,int] mapping TL4 unique_id -> TL3 index (0..n_tl3-1)
      n_tl3: int number of TL3 categories
    """
    # 1) EXACT same TL3 order as training: pandas unique() order (no sort!)
    tl3_unique = df_taxonomy["Type Level 3"].unique()
    level2int = {level: i for i, level in enumerate(tl3_unique)}
    int2skills_tl3 = {i: level for level, i in level2int.items()}
    n_tl3 = len(level2int)

    # 2) EXACT same mapping: unique_id -> Type Level 3
    skills_dict = dict(zip(df_taxonomy["unique_id"], df_taxonomy["Type Level 3"]))

    # 3) EXACT same mapping: unique_id -> tl3_index
    skills2int_tl3 = {int(uid): int(level2int[tl3]) for uid, tl3 in skills_dict.items()}

    return skills2int_tl3, int2skills_tl3, n_tl3

def initialize_all_data(canonical_col: str = "Type Level 4") -> Dict[str, Any]:
    """
    Load all data required by the chatbot at startup:
    - taxonomy CSV
    - alias → canonical mappings
    - canonical → unique_id mappings (TL4 skill IDs)
    - unique_id → canonical reverse mapping
    - job definitions
    - mastery level definitions
    - complete pool of skill IDs
    """
    # Load taxonomy once
    df_taxonomy = pd.read_csv(TAXONOMY_CSV)

    # Alias and canonical mappings
    alias2canon, canon2uid_full = build_alias_index(df_taxonomy, canonical_col)


    ############################################################################################
    # Simple canonical → uid map based directly on TL4
    #canon2uid = canon_to_unique_id_map(df_taxonomy, canonical_col)

    # Reverse map uid → canonical name
    #uid2canon = {uid: canon for canon, uid in canon2uid.items()}
    ############################################################################################
    canon2uid, uid2canon = canon_to_uid_maps(df_taxonomy, canonical_col)


    # Mapping to Type Level 3 for compatibility with RL agent
    skills2int_tl3, int2skills_tl3, n_tl3 = build_skills2int_tl3_from_taxonomy(df_taxonomy)

    # Runtime job, level data, courses
    jobs = load_json(JOBS_JSON)
    jobs = subsample_jobs_like_dataset(jobs, nb_jobs=-1, seed=42)
    levels = load_json(LEVELS_JSON)
    courses = load_json(COURSES_JSON)

    courses_requirements = {}
    courses_acquisitions = {}

    for course in courses:
        requirements = courses[course].get("required", [])
        acquisitions = courses[course].get("to_acquire", [])
        courses_requirements[course] = requirements
        courses_acquisitions[course] = acquisitions


    # Flat list of all TL4 unique_ids as strings
    #skills_pool = [str(uid) for uid in canon2uid.values()]
    skills_pool = []
    for v in canon2uid.values():
        if isinstance(v, list):
            skills_pool.extend(str(u) for u in v)
        else:
            skills_pool.append(str(v))

    return {
        "df_taxonomy": df_taxonomy,
        "alias2canon": alias2canon,
        "canon2uid": canon2uid,
        "uid2canon": uid2canon,
        "jobs": jobs,
        "levels": levels,
        "courses_requirements": courses_requirements,
        "courses_acquisitions": courses_acquisitions,
        "skills_pool": skills_pool,
        "skills2int_tl3": skills2int_tl3,
        "n_tl3": n_tl3,
        "int2skills_tl3": int2skills_tl3,
    }
