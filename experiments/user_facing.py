from ollama import chat


# Generates a user-friendly response using the original query
# and products retrieved from ChromaDB
def generate_response(user_query, results):

    product_information = ""

    for i in range(len(results["ids"][0])):

        product_id = results["ids"][0][i]
        product_text = results["documents"][0][i]
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]

        product_information += f"""
Product {i + 1}
Product ID: {product_id}
Distance: {distance:.3f}
Price: ${metadata["price_usd"]}
Rating: {metadata["rating"]}
Reviews: {metadata["reviews"]}
Product information: {product_text}
"""

    response = chat(
        model="llama3.1:latest",
        messages=[
            {
                "role": "system",
                "content": """
You are a concise beauty product recommendation assistant.

Answer the user's request using only the retrieved product information.

Recommend the strongest matching product or products.
Briefly explain why each recommendation matches the user's preferences.

Do not mention:
- the search process
- the retrieved products as a group
- product ranking or result numbers
- other products that were not recommended
- how you selected the products
- internal reasoning

Do not say that you "searched", "found", or "retrieved" products.

Do not invent any information that is not present in the product data.

Keep the response natural, concise, and user-facing.
"""
            },
            {
    "role": "user",
    "content": f"""
User request:
{user_query}

Product information:
{product_information}

Give the user a direct recommendation based on this information.
"""
            }
        ]
    )

    return response.message.content


# Test data that mimics the structure returned by ChromaDB
test_results = {
    "ids": [[
        "P483080",
        "P395032",
        "P502260"
    ]],

    "documents": [[
        "Product: Crescent Beach Perfume Travel Spray | "
        "Brand: ABBOTT | "
        "Category: Fragrance > Women > Rollerballs & Travel Size | "
        "Highlights: ['Vegan', 'Fresh Scent', 'Cruelty-Free'] | "
        "Variation: 0.33 oz / 10 mL eau de parfum spray",

        "Product: Book Expressive Travel Spray | "
        "Brand: Commodity | "
        "Category: Fragrance > Women > Rollerballs & Travel Size | "
        "Highlights: ['Vegan', 'Unisex/ Genderless Scent', "
        "'Clean at Sephora', 'Cruelty-Free'] | "
        "Variation: 0.34 oz / 10 mL perfume spray",

        "Product: Book+ Bold Travel Spray | "
        "Brand: Commodity | "
        "Category: Fragrance > Women > Rollerballs & Travel Size | "
        "Highlights: ['Vegan', 'Unisex/ Genderless Scent', "
        "'Clean at Sephora', 'Cruelty-Free'] | "
        "Variation: 0.34 oz / 10 mL perfume spray"
    ]],

    "distances": [[
        0.368,
        0.379,
        0.393
    ]],

    "metadatas": [[
        {
            "price_usd": 29.0,
            "rating": 4.7778,
            "reviews": 9.0
        },
        {
            "price_usd": 25.0,
            "rating": 3.9922,
            "reviews": 256.0
        },
        {
            "price_usd": 25.0,
            "rating": 5.0,
            "reviews": 1.0
        }
    ]]
}


# Test the final user-facing response
user_query = "I want a fresh travel-size perfume for women"

response = generate_response(user_query, test_results)

print("\nFINAL RESPONSE:")
print(response)