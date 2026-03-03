import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple, Set

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SkillSearcher:
    """Semantic nearest-neighbor over taxonomy embeddings (name+definition)."""

    def __init__(self, valid_uids, emb_path: str = "E_skills.npy", uids_path: str = "uids.npy",
                 model_name: str = MODEL_NAME, taxonomy_path: str = "./Data - Collection/Final/taxonomy.csv"):
        # Load precomputed embeddings and aligned uids
        self.E = np.load(emb_path).astype("float32")
        self.uids = np.load(uids_path).astype(int)
        self.model = SentenceTransformer(model_name)
        self.mu = self.E.mean(axis=0).astype("float32")
        self.mu /= (np.linalg.norm(self.mu) + 1e-9)
        self.df = pd.read_csv(taxonomy_path)
        self.cross_encoder_name = "cross-encoder/stsb-roberta-base" #cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.reranker = None

        full_map = self.df.set_index("unique_id")["name+definition"].to_dict()

        self.uid_to_text_map = {}

        for uid in full_map:
            if uid in valid_uids:
                self.uid_to_text_map[uid] = full_map[uid]

    def search(self, text: str, top_k: int = 30, min_sim: float = 0.35) -> List[Tuple[int, float]]:
        """Return top-k (uid_skill, cosine_sim) above threshold for the given text."""

        q = self.model.encode([text], normalize_embeddings=True)[0].astype("float32")  # [d]
        sims = self.E @ q
        if top_k >= len(sims):
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, top_k)[:top_k]
            idx = idx[np.argsort(-sims[idx])]
        return [(int(self.uids[i]), float(sims[i])) for i in idx if sims[i] >= min_sim]

    def ids(self, text: str, top_k: int = 30, min_sim: float = 0.35) -> Set[str]:
        """:returns set of skills ids as strings (for jobs.json filtering)."""
        return {str(uid) for uid, _ in self.search(text, top_k=top_k, min_sim=min_sim)}

    def search_reranked(self, text: str, top_k: int = 40, min_ce: float = 0.35) -> list[tuple[str, float]]:
        """Simplified reranking: retrieve with bi-encoder, rerank with cross-encoder."""
        # Step 1: normal search to get candidates
        candidates = self.search(text, top_k=top_k, min_sim=0.60)
        #print(candidates)
        if not candidates:
            return []

        # Step 2: load the cross-encoder (once)
        reranker = self._get_reranker()

        # Step 3: build (query, skill_text) pairs
        pairs = []
        pair_uids = []

        for uid_int, _ in candidates:
            uid = str(uid_int)
            skill_text = str(self.uid_to_text_map.get(uid_int))
            #print(skill_text)
            if skill_text:
                pairs.append((text, skill_text))
                pair_uids.append(uid)

        # Step 4: calculate scores and return
        scores = reranker.predict(pairs)
        ranked = sorted(zip(pair_uids, scores), key=lambda x: x[1], reverse=True)
        filtered = [(u, s) for (u, s) in ranked if s >= min_ce]
        return filtered

    def _get_reranker(self):
        if self.cross_encoder_name is None:
            return None
        if self.reranker is None:
            self.reranker = CrossEncoder(self.cross_encoder_name)
        return self.reranker
