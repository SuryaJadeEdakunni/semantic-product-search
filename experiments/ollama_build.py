from ollama import chat
from pydantic import BaseModel


# Structure returned by Ollama
class SearchRequirements(BaseModel):
    semantic_query: str


# Takes the user's natural-language request and extracts
# the descriptive/product preferences
def understand_query(user_query):

    response = chat(
        model="llama3.1:latest",
        messages=[
            {
                "role": "system",
                "content": """
You are a product search assistant.

Your ONLY task is to extract the descriptive and product preferences
that the user explicitly stated.

Return one field called semantic_query.

WHAT TO INCLUDE:
Include descriptive preferences that the user actually mentioned,
such as:
- luxurious
- feminine
- fresh
- elegant
- not too strong
- for summer
- travel-size
- vegan
- highly rated
- woody
- earthy
- floral
- beachy

WHAT TO REMOVE:
Remove explicit numerical constraints from the semantic query.

Examples:
- "under $50" -> remove from semantic_query
- "below $30" -> remove from semantic_query
- "at least 4.5 stars" -> remove from semantic_query
- "at least 100 reviews" -> remove from semantic_query

IMPORTANT:
Do NOT add preferences that the user did not state.

Do NOT infer:
- "5 stars" means feminine
- "5 stars" means luxurious
- "5 reviews" means popular
- "many reviews" means high quality
- a high rating means a product is feminine
- a low price means a product is cheap in style or luxurious in appearance
- "summer" means beach
- "feminine" means floral
- "luxurious" means expensive
- "travel-size" means suitable for women
- "vegan" means cruelty-free unless the product information later confirms this

The semantic query should represent WHAT THE USER ASKED FOR,
not what you think would satisfy the user.

EXAMPLES:

Example 1:
User:
"I want a vegan perfume under $40 with a rating of at least 4.5."

Correct semantic_query:
"vegan perfume"

Example 2:
User:
"I want something luxurious and feminine for summer, but I don't want
to spend more than $50 and I want something highly rated."

Correct semantic_query:
"luxurious feminine summer highly rated"

Do NOT change this to:
"luxurious feminine summer travel-size"
because the user did not mention travel-size.

Do NOT change this to:
"luxurious feminine floral summer"
because the user did not mention floral.

Example 3:
User:
"I want a fresh travel-size perfume for women that would be good
for a beach vacation."

Correct semantic_query:
"fresh travel-size perfume for women beach vacation"

Example 4:
User:
"I need an elegant perfume for my girlfriend's birthday that isn't too strong."

Correct semantic_query:
"elegant perfume girlfriend's birthday not too strong"

Do NOT add:
"floral"
"feminine"
"romantic"
unless the user explicitly says those things.

Example 5:
User:
"I want something cheap that smells expensive."

Correct semantic_query:
"cheap smells expensive"

Do NOT turn this into:
"luxurious perfume"
unless the user actually said luxurious.

IMPORTANT:
Do not use product information, ratings, review counts, or retrieved
products to interpret the user's request. Your job is ONLY to process
the user's original sentence.

Return ONLY the semantic_query.
"""
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        format=SearchRequirements.model_json_schema()
    )

    requirements = SearchRequirements.model_validate_json(
        response.message.content
    )

    return requirements


# Continuously accepts user queries
while True:

    try:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")

        if user_query.lower() == "exit":
            break

        requirements = understand_query(user_query)

        print("\nSemantic query:", requirements.semantic_query)

    except Exception as e:
        print(f"An error occurred: {e}")
##Queries with logical constraints

# I want a vegan perfume under $40 with a rating of at least 4.5       #Picked up on semantic, max price, min rating
# Find me a travel-size perfume under $30 with at least 50 reviews     #Same here


##Queries without logical constraints

# I want a fresh and beachy perfume for women #Understood the semantic and returned none for the logical which is good 
# I am looking for a woody and earthy fragrance for an evening out #Same here 


##Complex queries

# I need an elegant perfume for my girlfriend's birthday that is fresh, not too strong, and preferably under $60
# I want something luxurious and feminine for summer, but I don't want to spend more than $50 and I want something highly rated
