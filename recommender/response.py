"""Stage 3: turn retrieved candidates into 1-3 grounded, user-facing recommendations."""

from ollama import chat

from recommender.config import LLM_MODEL

SYSTEM_PROMPT = """
You are a beauty product recommendation assistant.

Select the 1-3 strongest products from the retrieved candidates that best
match the user's request. Use lower semantic distance as one relevance signal.

For each recommendation, explain specifically why it matches by referring
to evidence in the provided product information, such as the description,
highlights, category, ingredients, or variation. If the relevant information
is not obvious from the product name, explain where the match comes from
naturally.

Only make claims supported by the provided product information.
Do not invent product details or characteristics.
Ratings only indicate rating, and review counts do not prove popularity.
Do not make unsupported medical or safety claims.
Do not use an unrelated characteristic as evidence for another requirement.

Do not mention semantic distance, embeddings, vector databases, retrieval,
ranking, or internal reasoning.

Present each recommendation as:

### Product Name
**Brand:**
**Price:**
**Rating:**
**Reviews:**
**Category:**
**Highlights:**
**Size:**
**Why it matches:**

Include ingredients only when relevant.
Keep the response concise, natural, and user-facing.
"""


def _format_candidates(results) -> str:
    product_information = ""

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]

        product_information += f"""
Candidate:
Product ID: {results["ids"][0][i]}
Semantic distance: {results["distances"][0][i]:.3f}
Price: ${metadata["price_usd"]}
Rating: {metadata["rating"]}
Reviews: {metadata["reviews"]}
Product information: {results["documents"][0][i]}
"""
    return product_information


def generate_response(user_query: str, results) -> str:
    """Ask the LLM to recommend the best matches using only the retrieved information."""

    response = chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
User request:
{user_query}

Retrieved product candidates:
{_format_candidates(results)}

Recommend the strongest matches using only the retrieved information.
""",
            },
        ],
    )

    return response.message.content
