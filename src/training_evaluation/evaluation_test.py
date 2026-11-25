
# 1. Install dependencies
!pip install -q -U transformers peft bitsandbytes accelerate sentencepiece scikit-learn

import json
import random
import re
import torch
from collections import Counter
from google.colab import drive

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# 2. Mount Google Drive
drive.mount("/content/drive")

# 3. Paths and settings
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

PROJECT_DIR = "/content/drive/MyDrive/Colab_Notebooks/Fine_Tuning_Lora"

DATA_PATH = f"{PROJECT_DIR}/test.jsonl"
ADAPTER_PATH = f"{PROJECT_DIR}/lora-adapter"
OUTPUT_PATH = f"{PROJECT_DIR}/test_evaluation_report.txt"

LABELS = ["Exact", "Substitute", "Complement", "Irrelevant"]
SAMPLE_SIZE = 2000
SEED = 42

# 4. Randomly sample 2,000 examples from the full test set
rng = random.Random(SEED)
records = []
total_rows = 0

with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        if row["response"] not in LABELS:
            raise ValueError(f"Unexpected label: {row['response']}")

        total_rows += 1

        if len(records) < SAMPLE_SIZE:
            records.append(row)
        else:
            j = rng.randrange(total_rows)
            if j < SAMPLE_SIZE:
                records[j] = row

if len(records) < SAMPLE_SIZE:
    raise ValueError(f"Test set contains only {len(records)} examples")

rng.shuffle(records)

print("Full test set:", total_rows)
print("Sampled test examples:", len(records))
print("Sample distribution:", Counter(r["response"] for r in records))

# 5. Load tokenizer and quantized base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

base_model.eval()

# 6. Prediction function
def predict(model, row):
    prompt = (
        "<s>[INST] "
        + row["instruction"]
        + "\n\n"
        + row["input"]
        + " [/INST]"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    match = re.match(
        r"^(Exact|Substitute|Complement|Irrelevant)\b",
        text,
        re.IGNORECASE,
    )

    return match.group(1).capitalize() if match else "INVALID"

# 7. Evaluation function
def evaluate(model, name):
    y_true = []
    y_pred = []

    for i, row in enumerate(records, 1):
        y_true.append(row["response"])
        y_pred.append(predict(model, row))

        if i % 100 == 0:
            print(f"{name}: {i}/{len(records)}")

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        digits=4,
    )

    return (
        f"\n=== {name} ===\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro-F1: {macro_f1:.4f}\n"
        f"Invalid predictions: {y_pred.count('INVALID')}\n"
        f"{report}\n"
    )

# 8. Evaluate base model
results = evaluate(base_model, "BASE MODEL")

# 9. Evaluate base + LoRA on the same sample
lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

lora_model.eval()

results += evaluate(lora_model, "BASE + LoRA")

# 10. Display and save results
print(results)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Full test set size: {total_rows}\n")
    f.write(f"Sample size: {SAMPLE_SIZE}\n")
    f.write(f"Sample distribution: {Counter(r['response'] for r in records)}\n")
    f.write(results)

print("Report saved to:", OUTPUT_PATH)
