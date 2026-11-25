
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

SPLITS = ["train", "validation", "test"]
LABELS = {"Exact", "Substitute", "Complement", "Irrelevant"}
TARGET = 2500


# 1. Check query and product overlap
frames = {
    split: pd.read_parquet(DATA / f"esci_{split}.parquet")
    for split in SPLITS
}

for column in ["query_id", "product_id"]:
    sets = {
        split: set(df[column].dropna().astype(str))
        for split, df in frames.items()
    }

    for a, b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = len(sets[a] & sets[b])

        print(f"{column} overlap {a}/{b}: {overlap:,}")

        if column == "query_id" and overlap:
            raise ValueError("Query leakage detected.")


# 2. Reservoir sampling: 2,500 examples per class
rng = random.Random(42)

reservoirs = {label: [] for label in LABELS}
seen = {label: 0 for label in LABELS}

source = DATA / "train.jsonl"
output = DATA / "train_balanced.jsonl"

with source.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        label = row["response"]

        if label not in LABELS:
            continue

        seen[label] += 1
        count = seen[label]

        if len(reservoirs[label]) < TARGET:
            reservoirs[label].append(row)
        else:
            index = rng.randrange(count)

            if index < TARGET:
                reservoirs[label][index] = row

if any(seen[label] < TARGET for label in LABELS):
    raise ValueError("Not enough examples for balanced sampling.")

# 3. Save balanced training data
with output.open("w", encoding="utf-8") as f:
    for label in sorted(LABELS):
        for row in reservoirs[label]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\nOriginal training class counts:", seen)
print("Balanced training counts:", {
    label: len(reservoirs[label]) for label in LABELS
})
print("Total balanced examples:", TARGET * len(LABELS))
print("Saved:", output)

