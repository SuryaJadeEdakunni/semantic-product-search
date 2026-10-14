"""Shared settings for the recommender pipeline."""

import os
from pathlib import Path

# Repository root (one level above this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the persistent ChromaDB database.
# Override with the CHROMA_PATH environment variable if stored elsewhere.
CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", PROJECT_ROOT / "chroma_db"))

COLLECTION_NAME = "SephoraProducts"

# Must be the same model used to embed the products in scripts/build_index.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Local LLM served by Ollama, used for query interpretation and response generation
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:latest")

# Number of candidates retrieved from ChromaDB before the LLM picks the final 1-3
NUMBER_OF_RESULTS = 10
