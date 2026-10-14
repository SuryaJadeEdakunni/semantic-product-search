"""Stage 1: use a local LLM to extract the descriptive meaning of a user's request."""

from ollama import chat
from pydantic import BaseModel

from recommender.config import LLM_MODEL


# Structure returned by Ollama
class SearchRequirements(BaseModel):
    semantic_query: str


SYSTEM_PROMPT = """
You are a product search assistant.
Extract only the descriptive product preferences stated by the user.

Keep descriptive terms such as style, scent, size, use case,
product type, and qualitative preferences.

Do not add or infer information that the user did not provide.
Do not turn vague terms such as "cheap", "popular", or "highly rated"
into numerical thresholds.

Remove explicit numerical price, rating, and review constraints.

Return only semantic_query.
"""


def understand_query(user_query: str) -> str:
    """Return the descriptive search phrase extracted from a natural-language request."""

    response = chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        format=SearchRequirements.model_json_schema(),
    )

    requirements = SearchRequirements.model_validate_json(response.message.content)

    return requirements.semantic_query
