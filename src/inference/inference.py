
import re
import torch

from configs.model_config import MAX_NEW_TOKENS
from src.inference_deployment.load_model import load_lora_model


model, tokenizer = load_lora_model()

# Use the exact instruction from your training JSONL.
INSTRUCTION = (
    "Classify the relevance of the product to the shopping query. "
    "Choose exactly one label: Exact, Substitute, Complement, or Irrelevant."
)


def predict(query, title, brand=None, description=None, features=None):

    product = f"Title: {title}"

    if brand:
        product += f"\nBrand: {brand}"

    if description:
        product += f"\nDescription: {description[:500]}"

    if features:
        product += f"\nFeatures: {features[:500]}"

    input_text = f"Query: {query}\nProduct:\n{product}"

    prompt = (
        f"<s>[INST] {INSTRUCTION}\n\n"
        f"{input_text} [/INST]"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.get_input_embeddings().weight.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    match = re.match(
        r"^(Exact|Substitute|Complement|Irrelevant)\b",
        answer,
        re.IGNORECASE,
    )

    return match.group(1).capitalize() if match else "INVALID"


if __name__ == "__main__":

    prediction = predict(
        query="wireless noise cancelling headphones",
        title="Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        brand="Sony",
        description="Wireless headphones with active noise cancellation.",
    )

    print("Predicted ESCI label:", prediction)

