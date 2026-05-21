import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer

INPUT_FILE = "candidate_pairs_with_features.csv"
OUTPUT_FILE = "pairs_with_sbert.csv"
MODEL_NAME = "all-MiniLM-L6-v2"


def cosine_similarity_batch(vecs_a, vecs_b):
    norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    vecs_a_norm = vecs_a / np.maximum(norms_a, 1e-10)
    vecs_b_norm = vecs_b / np.maximum(norms_b, 1e-10)
    return np.sum(vecs_a_norm * vecs_b_norm, axis=1).clip(0, 1)


def prepare_text(row, suffix):
    # Build the COMBINED_TEXT representation for one record in a pair
    def get(col_lower, col_upper):
        for key in [col_lower, col_upper, col_lower.upper(), col_upper.lower()]:
            if key in row.index and pd.notna(row[key]):
                return str(row[key])
        return ""

    cae = get(f"cae_name_{suffix}", f"CAE_NAME_{suffix}")
    win = get(f"win_name_{suffix}", f"WIN_NAME_{suffix}")
    title = get(f"title_{suffix}", f"TITLE_{suffix}")
    parts = [p for p in [cae, win, title] if p.strip()]
    return " ".join(parts).lower().strip()


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} pairs")

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    texts_a = [prepare_text(row, "1") for _, row in df.iterrows()]
    texts_b = [prepare_text(row, "2") for _, row in df.iterrows()]

    # Replace empty texts with placeholder so the encoder does not crash
    texts_a = [t if t else "empty" for t in texts_a]
    texts_b = [t if t else "empty" for t in texts_b]

    t0 = time.time()
    all_embeddings = model.encode(
        texts_a + texts_b,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    n = len(df)
    similarities = cosine_similarity_batch(all_embeddings[:n], all_embeddings[n:])
    print(f"Encoding finished in {time.time() - t0:.1f}s")

    df["sbert_similarity"] = similarities
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"sbert_similarity: min={similarities.min():.3f}, "
          f"max={similarities.max():.3f}, mean={similarities.mean():.3f}")


if __name__ == "__main__":
    main()
