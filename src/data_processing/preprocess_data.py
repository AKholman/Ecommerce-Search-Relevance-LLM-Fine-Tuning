
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "esci"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def assign_split(query_id):
    h = int(
        hashlib.sha256(str(query_id).encode()).hexdigest()[:8],
        16
    ) % 10

    return "validation" if h == 0 else "train"


# 1. Load raw ESCI data
examples = pd.read_parquet(
    RAW / "shopping_queries_dataset_examples.parquet"
)

products = pd.read_parquet(
    RAW / "shopping_queries_dataset_products.parquet"
)

# 2. Filter US data
examples = examples[
    examples["product_locale"] == "us"
].copy()

products = products[
    products["product_locale"] == "us"
].copy()

# 3. Check product uniqueness
if products.duplicated(["product_id", "product_locale"]).any():
    raise ValueError("Duplicate US product IDs detected.")

# 4. Identify queries appearing in multiple original splits
query_splits = examples[
    ["query_id", "split"]
].drop_duplicates()

split_counts = query_splits.groupby("query_id")["split"].nunique()

conflicting_queries = split_counts[split_counts > 1].index

print("Conflicting queries:", len(conflicting_queries))

# 5. Assign one split per query
original = (
    query_splits.drop_duplicates("query_id")
    .set_index("query_id")["split"]
    .astype(str).str.lower()
)

if not original.isin(["train", "test"]).all():
    raise ValueError("Unexpected original split values.")

# Conflicting queries go entirely to the training side
original.loc[conflicting_queries] = "train"

# Split official training queries into 90% train / 10% validation
new_splits = original.copy()

train_ids = original[original == "train"].index

new_splits.loc[train_ids] = [
    assign_split(query_id) for query_id in train_ids
]

# Apply the same split to every example belonging to each query
examples["split"] = examples["query_id"].map(new_splits)

# 6. Join examples with product metadata
merged = examples.merge(
    products,
    on=["product_id", "product_locale"],
    how="left",
    validate="many_to_one",
    suffixes=("", "_product")
)

missing = merged["product_title"].isna().sum()
print("Missing product metadata:", missing)

if missing:
    raise ValueError("Some query-product pairs have no product metadata.")

# 7. Save the three datasets
for split in ["train", "validation", "test"]:
    part = merged[merged["split"] == split].copy()

    part.to_parquet(
        OUT / f"esci_{split}.parquet",
        index=False
    )

    print(f"ESCI {split}: {len(part):,} rows")

print("ESCI preprocessing complete.")
