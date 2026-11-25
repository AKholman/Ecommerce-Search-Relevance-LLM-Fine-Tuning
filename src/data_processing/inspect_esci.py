
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "esci"

examples = pd.read_parquet(
    RAW / "shopping_queries_dataset_examples.parquet"
)

products = pd.read_parquet(
    RAW / "shopping_queries_dataset_products.parquet"
)

examples_us = examples[
    examples["product_locale"] == "us"
]

products_us = products[
    products["product_locale"] == "us"
]

print("Original examples:", len(examples))
print("Original unique queries:", examples["query_id"].nunique())
print("Original products:", len(products))

print("\nUS examples:", len(examples_us))
print("US unique queries:", examples_us["query_id"].nunique())
print("US products:", len(products_us))
print("US unique products:", products_us["product_id"].nunique())

print("\nESCI label counts:")
print(examples_us["esci_label"].value_counts().sort_index())

print("\nESCI label percentages:")
print(
    examples_us["esci_label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nOriginal US splits:")
print(examples_us["split"].value_counts())
