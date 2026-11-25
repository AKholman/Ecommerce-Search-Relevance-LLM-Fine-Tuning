
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

SPLITS = ["train", "validation", "test"]

LABELS = {
    "E": "Exact",
    "S": "Substitute",
    "C": "Complement",
    "I": "Irrelevant",
}

INSTRUCTION = (
    "Classify the relevance of the product to the shopping query. "
    "Choose exactly one label: Exact, Substitute, Complement, or Irrelevant."
)


def clean(value):
    return str(value).strip() if pd.notna(value) else ""


def main():
    for split in SPLITS:
        source = DATA / f"esci_{split}.parquet"
        output = DATA / f"{split}.jsonl"

        if not source.exists():
            raise FileNotFoundError(source)

        df = pd.read_parquet(source)

        count = 0

        with output.open("w", encoding="utf-8") as f:
            for row in df.itertuples(index=False):
                query = clean(row.query)
                title = clean(row.product_title)
                label = clean(row.esci_label).upper()

                if not query or not title or label not in LABELS:
                    continue

                product = f"Title: {title}"

                brand = clean(row.product_brand)
                description = clean(row.product_description)
                bullets = clean(row.product_bullet_point)

                if brand:
                    product += f"\nBrand: {brand}"

                if description:
                    product += f"\nDescription: {description[:500]}"

                if bullets:
                    product += f"\nFeatures: {bullets[:500]}"

                record = {
                    "id": f"esci_{split}_{count:07d}",
                    "instruction": INSTRUCTION,
                    "input": f"Query: {query}\nProduct:\n{product}",
                    "response": LABELS[label],
                }

                f.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                count += 1

        print(f"{split}: {count:,} examples -> {output}")


if __name__ == "__main__":
    main()
