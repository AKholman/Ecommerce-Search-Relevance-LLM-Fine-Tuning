
# Install dependencies in Colab first:
# !pip install -U transformers datasets peft accelerate bitsandbytes sentencepiece

import json
import torch
from pathlib import Path
from google.colab import drive

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# 1. Mount Google Drive
drive.mount("/content/drive")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/Fine_Tuning_Lora/train_balanced.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/Colab_Notebooks/Fine_Tuning_Lora/lora-adapter"
MAX_LENGTH = 512

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# 4. Load 40,000 balanced examples
records = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

assert len(records) == 10_000, f"Expected 10,000 rows, got {len(records)}"

# 5. Format prompts; calculate loss only on the answer
def preprocess(example):
    prompt = (
        "<s>[INST] " + example["instruction"] + "\n\n"
        + example["input"] + " [/INST]"
    )
    answer = " " + example["response"] + tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]
    answer_ids = tokenizer(
        answer, add_special_tokens=False
    )["input_ids"]

    prompt_ids = prompt_ids[:MAX_LENGTH - len(answer_ids)]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return {"input_ids": input_ids, "labels": labels}

dataset = Dataset.from_list(records)
tokenized = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    desc="Tokenizing examples",
)

# 6. Configure rank-16 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 7. Train
training_args = TrainingArguments(
    output_dir="/content/lora-training-checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-4,
    warmup_steps=70,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
    seed=42,
    data_seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    ),
)

trainer.train()

# 8. Save adapter and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapter saved to: {OUTPUT_DIR}")