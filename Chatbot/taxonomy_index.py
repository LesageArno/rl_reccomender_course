import re
from difflib import get_close_matches
from typing import Dict, Tuple, List

import pandas as pd

from .utils import normalize, tokenize


LEVEL_COLS = ["Type Level 1", "Type Level 2", "Type Level 3", "Type Level 4"]


def n_grams(tokens: List[str], n: int) -> List[str]:
    """Return all contiguous n-grams from a token sequence."""
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def build_alias_index(
    df_taxonomy: pd.DataFrame,
    name_column: str = "Type Level 4",
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Build two lookup structures from the taxonomy:

    - alias2canon: any alias (names, TL1..TL4, altLabels) → canonical name (normalized)
    - canon2uid : canonical name → unique_id

    The canonical name is taken from `name_column` and normalized.
    """
    df = df_taxonomy

    alias2canon: Dict[str, str] = {}
    canon2uid: Dict[str, int] = {}

    for _, row in df.iterrows():
        uid = row.get("unique_id")
        if pd.isna(uid):
            continue
        uid = int(uid)

        name_raw = str(row.get(name_column, "")).strip()
        if not name_raw:
            continue
        canonical_name = normalize(name_raw)

        # register canonical name → uid (first occurrence wins)
        canon2uid.setdefault(canonical_name, uid)

        # collect all potential aliases: TL1..TL4, altLabels, the TL4 name itself
        aliases = set()

        tl4_raw = str(row.get("Type Level 4", "")).strip()
        if tl4_raw:
            aliases.add(normalize(tl4_raw))

        for col in LEVEL_COLS:
            if col in df.columns:
                val = str(row.get(col, "")).strip()
                if val:
                    aliases.add(normalize(val))

        alts_raw = str(row.get("altLabels", "") or "")
        if alts_raw:
            for chunk in re.split(r"[|;,]", alts_raw):
                chunk = chunk.strip()
                if chunk:
                    aliases.add(normalize(chunk))

        aliases.add(canonical_name)

        # direct aliases
        for alias in aliases:
            if alias:
                alias2canon.setdefault(alias, canonical_name)

        # n-gram aliases (bigrams / trigrams)
        for phrase in list(aliases):
            tokens = [token for token in tokenize(phrase)]
            for bg in n_grams(tokens, 2):
                alias2canon.setdefault(bg, canonical_name)
            for tg in n_grams(tokens, 3):
                alias2canon.setdefault(tg, canonical_name)

    return alias2canon, canon2uid


def alias_lookup(
    token: str,
    alias2canon: Dict[str, str],
    cutoff: float = 0.8,
) -> str | None:
    """
    Resolve a token to its canonical form using:

    1. Exact alias match
    2. Fuzzy match (difflib) if no exact match is found
    """
    block = {
        "and", "or", "not", "i", "prefer", "like", "love",
        "want", "to", "a", "an", "the",
    }
    if token in block:
        return None

    local_cutoff = cutoff
    if 5 <= len(token) <= 8 and token.isalpha():
        local_cutoff = 0.75

    if token in alias2canon:
        return alias2canon[token]

    candidates = get_close_matches(
        token,
        alias2canon.keys(),
        n=1,
        cutoff=local_cutoff,
    )
    return alias2canon[candidates[0]] if candidates else None
