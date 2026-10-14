import chromadb
from sentence_transformers import SentenceTransformer

# Loading the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connects to the database previously created in chroma_setup.py
client = chromadb.PersistentClient(path = "./chroma_db")

# Loading existing collection of products from the database
products = client.get_collection(name = "SephoraProducts")


# Search function using semantic search + numerical metadata filtering
def search_products(query, max_price = None, min_rating = None, min_reviews = None, number_of_results = 5):

    # Turning the user's text into a 384-dimensional embedding
    query_embedding = model.encode(query).tolist()

    # Creating numerical filters if the user provides them
    filters = []

    if max_price is not None:
        filters.append({"price_usd": {"$lte": max_price}})

    if min_rating is not None:
        filters.append({"rating": {"$gte": min_rating}})

    if min_reviews is not None:
        filters.append({"reviews": {"$gte": min_reviews}})

    # Combining multiple numerical conditions
    if len(filters) == 0:
        where_filter = None
    elif len(filters) == 1:
        where_filter = filters[0]
    else:
        where_filter = {"$and": filters}

    # Searching ChromaDB using semantic similarity
    # and applying numerical metadata filters when provided
    if where_filter is None:
        results = products.query(
            query_embeddings = [query_embedding],
            n_results = number_of_results,
            include = ["documents", "distances", "metadatas"]
        )
    else:
        results = products.query(
            query_embeddings = [query_embedding],
            n_results = number_of_results,
            where = where_filter,
            include = ["documents", "distances", "metadatas"]
        )

    print()
    print(f"QUERY: {query}")

    for i in range(len(results["ids"][0])):

        product_id = results["ids"][0][i]
        product_text = results["documents"][0][i]
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]
    
        print()
        print(f"RESULT {i + 1}")
        print(f"Product ID: {product_id}")
        print(f"Distance: {distance:.3f}")
        print(f"Price: ${metadata['price_usd']}")
        print(f"Rating: {metadata['rating']}")
        print(f"Reviews: {metadata['reviews']}")
        print(f"Product: {product_text}")


# Continuously accepts user queries
while True:

    try:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")

        if user_query.lower() == "exit":
            break

        search_products(user_query)

    except Exception as e:
        print(f"An error occurred: {e}")
        break

##Queries Checked : Simple / Clear

# vegan perfume for women

# woody earthy scented perfume


##Queries Checked : Complex / Natural Language

# I want a fresh travel-size perfume for women that would be good for a beach vacation

# I need an elegant perfume for my girlfriend's birthday that isn't too strong


##Queries Checked : Difficult / Poor Semantic Match

# I want something cheap that smells expensive

# I need a product that won't irritate my sensitive skin