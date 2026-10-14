"""
Step 2 of rebuilding the database: load the processed data into a ChromaDB collection.

Requires the outputs of scripts/build_dataset.py.

Run from the project root:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from recommender.config import (  # noqa: E402
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BATCH_SIZE = 5000  # ChromaDB's maximum batch size


def main():
    product_embeddings = np.load(PROCESSED_DIR / "product_embeddings.npy")
    metadata_df = pd.read_pickle(PROCESSED_DIR / "metadata_df.pkl")
    semantic_df = pd.read_pickle(PROCESSED_DIR / "semantic_df.pkl")

    assert len(product_embeddings) == len(metadata_df) == len(semantic_df), \
        "Processed files are out of sync; re-run build_dataset.py"

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Start from a clean collection (safe on a first run when none exists yet)
    existing = [c.name if hasattr(c, "name") else c for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(name=COLLECTION_NAME)

    products = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    product_ids = metadata_df["product_id"].tolist()
    product_texts = semantic_df["product_text"].tolist()
    # Numerical metadata used for filtering (price, rating, review count)
    product_metadata = (
        metadata_df[["price_usd", "rating", "reviews"]].fillna(0).to_dict(orient="records")
    )

    for start in range(0, len(product_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(product_ids))
        products.add(
            ids=product_ids[start:end],
            embeddings=product_embeddings[start:end].tolist(),
            documents=product_texts[start:end],
            metadatas=product_metadata[start:end],
        )
        print(f"Added products {start} to {end - 1}")

    print(f"Products in ChromaDB: {products.count():,}")

    # Sanity check with a natural-language query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query = "I want a fresh travel-size perfume for women"
    results = products.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=5,
    )
    print(f"\nTop matches for: '{query}'")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"  {dist:.3f}  {doc[:90]}...")


if __name__ == "__main__":
    main()
