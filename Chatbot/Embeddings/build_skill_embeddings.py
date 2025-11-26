import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def build_skill_embeddings(
    taxonomy_csv: str,
    text_col: str = "name+definition",   # your pre-made column
    uid_col: str = "unique_id",
    out_emb_path: str = "E_skills.npy",
    out_uid_path: str = "uids.npy",
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
):
    """Compute embeddings for the column containing name+definition."""
    df = pd.read_csv(taxonomy_csv)
    df = df[df[uid_col].notna()].copy()
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in {taxonomy_csv}")

    texts = df[text_col].fillna("").astype(str).tolist()
    uids = df[uid_col].astype(int).to_list()

    if "altLabels" in df.columns:
        for uid, blob in zip(df[uid_col].astype(int), df["altLabels"].fillna("").astype(str)):
            # One alt label per line; strip and skip empties
            for alt in [t.strip() for t in blob.splitlines() if t.strip()]:
                texts.append(alt)
                uids.append(uid)

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    Path(out_emb_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb_path, emb)
    np.save(out_uid_path, uids)

    print(f"✅ Saved {emb.shape[0]} embeddings ({emb.shape[1]} dims)")
    print(f"→ {out_emb_path}, {out_uid_path}")

if __name__ == "__main__":
    build_skill_embeddings("../Data-Collection/taxonomy.csv")
