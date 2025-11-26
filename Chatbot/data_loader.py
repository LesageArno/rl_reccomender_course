import json
import pandas as pd
from typing import Dict, Any

from .utils import normalize
from .taxonomy_index import build_alias_index

# Paths for chatbot runtime
TAXONOMY_CSV = "Data - Collection/Final/taxonomy.csv"
JOBS_JSON = "Data - Collection/Final/jobs.json"
LEVELS_JSON = "Data - Collection/Final/mastery_levels.json"


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return a Python dict."""
    with open(path, "r") as f:
        return json.load(f)


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

    # Simple canonical → uid map based directly on TL4
    canon2uid = canon_to_unique_id_map(df_taxonomy, canonical_col)

    # Reverse map uid → canonical name
    uid2canon = {uid: canon for canon, uid in canon2uid.items()}


    # Runtime job and level data
    jobs = load_json(JOBS_JSON)
    levels = load_json(LEVELS_JSON)

    # Flat list of all TL4 unique_ids as strings
    skills_pool = [str(uid) for uid in canon2uid.values()]

    return {
        "df_taxonomy": df_taxonomy,
        "alias2canon": alias2canon,
        "canon2uid": canon2uid,
        "uid2canon": uid2canon,
        "jobs": jobs,
        "levels": levels,
        "skills_pool": skills_pool,
    }
