# Experiments

Scripts used during development to test each pipeline stage in isolation before combining them in `main.py`. They are kept for transparency and are not needed to run the app.

| Script | Purpose |
|---|---|
| `search_check.py` | Interactive semantic search against ChromaDB with optional price / rating / review filters, used to inspect raw retrieval quality and distances |
| `ollama_build.py` | Earlier, example-heavy version of the query-interpretation prompt, used to test which inferences the LLM should and shouldn't make |
| `user_facing.py` | First version of the response-generation stage, tested on hard-coded ChromaDB-style results |

Run from the project root so the scripts can find `chroma_db/`, e.g. `python experiments/search_check.py`.
