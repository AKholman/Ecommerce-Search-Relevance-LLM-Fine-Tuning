
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "esci"

files = {
    "examples": "shopping_queries_dataset_examples.parquet",
    "products": "shopping_queries_dataset_products.parquet",
    "sources": "shopping_queries_dataset_sources.csv",
}

for name, filename in files.items():
    path = RAW / filename

    if not path.exists():
        raise FileNotFoundError(path)

    df = (
        pd.read_parquet(path)
        if filename.endswith(".parquet")
        else pd.read_csv(path)
    )

    print(f"\n--- {name.upper()} ---")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isna().sum())
    print(df.head(2).to_string(index=False))

# Required columns
required_examples = {
    "query", "query_id", "product_id",
    "product_locale", "esci_label", "split"
}
required_products = {
    "product_id", "product_title", "product_locale"
}

examples = pd.read_parquet(RAW / files["examples"])
products = pd.read_parquet(RAW / files["products"])

assert required_examples.issubset(examples.columns)
assert required_products.issubset(products.columns)

assert set(examples["esci_label"].dropna().unique()) <= {
    "E", "S", "C", "I"
}

print("\nRaw ESCI checks passed.")

