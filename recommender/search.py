"""Stage 2: embed the semantic query and retrieve similar products from ChromaDB."""

import chromadb
from sentence_transformers import SentenceTransformer

from recommender.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    NUMBER_OF_RESULTS,
)

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        if not (CHROMA_PATH / "chroma.sqlite3").exists():
            raise FileNotFoundError(
                f"No ChromaDB database found at '{CHROMA_PATH}'.\n"
                "Download chroma_db.zip from the GitHub Releases page and unzip it "
                "into the project root, or rebuild it with:\n"
                "    python scripts/build_dataset.py\n"
                "    python scripts/build_index.py"
            )
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def search_products(
    query,
    max_price=None,
    min_rating=None,
    min_reviews=None,
    number_of_results=NUMBER_OF_RESULTS,
):
    """
    Embeds a semantic query, applies optional numerical filters,
    and retrieves matching products from ChromaDB.
    """

    query_embedding = _get_model().encode(query).tolist()

    filters = []

    if max_price is not None:
        filters.append({"price_usd": {"$lte": max_price}})

    if min_rating is not None:
        filters.append({"rating": {"$gte": min_rating}})

    if min_reviews is not None:
        filters.append({"reviews": {"$gte": min_reviews}})

    if len(filters) == 0:
        where_filter = None
    elif len(filters) == 1:
        where_filter = filters[0]
    else:
        where_filter = {"$and": filters}

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": number_of_results,
        "include": ["documents", "distances", "metadatas"],
    }

    if where_filter is not None:
        query_args["where"] = where_filter

    return _get_collection().query(**query_args)
