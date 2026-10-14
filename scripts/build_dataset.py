"""
Step 1 of rebuilding the database: download, clean and embed the Sephora product data.

Outputs (written to data/processed/):
    semantic_df.pkl         text fields used for embedding
    metadata_df.pkl         numerical / categorical metadata
    product_embeddings.npy  one 384-dimensional embedding per product

Run from the project root:
    python scripts/build_dataset.py
"""

import os
import sys
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from recommender.config import EMBEDDING_MODEL, PROJECT_ROOT  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SEMANTIC_COLUMNS = [
    "product_id",
    "product_name",
    "brand_name",
    "ingredients",
    "highlights",
    "primary_category",
    "secondary_category",
    "tertiary_category",
    "variation_type",
    "variation_value",
]

METADATA_COLUMNS = [
    "product_id",
    "price_usd",
    "rating",
    "reviews",
    "loves_count",
    "size",
    "limited_edition",
    "new",
    "online_only",
    "out_of_stock",
    "sephora_exclusive",
    "child_count",
    "brand_id",
]


def load_raw_data() -> pd.DataFrame:
    """Download the Sephora dataset from Kaggle and load product_info.csv."""
    path = kagglehub.dataset_download("nadyinky/sephora-products-and-skincare-reviews")
    return pd.read_csv(os.path.join(path, "product_info.csv"))


def clean(data: pd.DataFrame) -> pd.DataFrame:
    products = data.copy()

    # Too many missing values; price_usd already captures price
    products = products.drop(columns=["sale_price_usd", "value_price_usd", "variation_desc"])
    # Too many missing values; child_count already present
    products = products.drop(columns=["child_max_price", "child_min_price"])

    # Missing reviews / loves are treated as zero (same 278 rows are missing in both)
    products["reviews"] = products["reviews"].fillna(0)
    products["loves_count"] = products["loves_count"].fillna(0)

    return products


def build_product_text(semantic_df: pd.DataFrame) -> pd.DataFrame:
    # Combine the category hierarchy into one string, e.g. "Fragrance > Women > Perfume"
    semantic_df["category_path"] = (
        semantic_df[["primary_category", "secondary_category", "tertiary_category"]]
        .fillna("")
        .astype(str)
        .apply(lambda row: " > ".join(value for value in row if value), axis=1)
    )

    # Main text that gets embedded
    semantic_df["product_text"] = (
        "Product: " + semantic_df["product_name"].fillna("")
        + " | Brand: " + semantic_df["brand_name"].fillna("")
        + " | Category: " + semantic_df["category_path"].fillna("")
        + " | Highlights: " + semantic_df["highlights"].fillna("")
        + " | Ingredients: " + semantic_df["ingredients"].fillna("")
        + " | Variation: " + semantic_df["variation_type"].fillna("")
        + " " + semantic_df["variation_value"].fillna("")
    )
    return semantic_df


def main():
    data = load_raw_data()
    print(f"Loaded {len(data):,} products with {data.shape[1]} columns")

    products = clean(data)
    semantic_df = build_product_text(products[SEMANTIC_COLUMNS].copy())
    metadata_df = products[METADATA_COLUMNS].copy()

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        semantic_df["product_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "product_embeddings.npy", embeddings)
    semantic_df.to_pickle(OUTPUT_DIR / "semantic_df.pkl")
    metadata_df.to_pickle(OUTPUT_DIR / "metadata_df.pkl")
    print(f"Saved processed data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
