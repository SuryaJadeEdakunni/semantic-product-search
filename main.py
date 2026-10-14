"""
Semantic Product Search: interactive command-line app.

Pipeline:
    1. Llama 3.1 (via Ollama) extracts the descriptive meaning of the request
    2. Sentence Transformers + ChromaDB retrieve the 10 most similar products
    3. Llama 3.1 selects and explains the 1-3 best matches

Run from the project root:
    python main.py
"""

from recommender.response import generate_response
from recommender.search import search_products
from recommender.understand_query import understand_query


def main():
    print("Semantic Product Search. Describe what you're looking for.")

    while True:
        try:
            user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()

            if user_query.lower() == "exit":
                break
            if not user_query:
                continue

            # Ollama interprets the user's request
            semantic_query = understand_query(user_query)

            # ChromaDB retrieves relevant products
            results = search_products(semantic_query)

            # Ollama generates the final response
            response = generate_response(user_query, results)

            print("\nFINAL RESPONSE:")
            print(response)

        except FileNotFoundError as e:
            print(f"\n{e}")
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
