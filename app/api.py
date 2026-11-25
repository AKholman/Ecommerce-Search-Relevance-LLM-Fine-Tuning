import re
import torch
from configs.model_config import MAX_NEW_TOKENS


def predict(data, tokenizer, model):

    instruction = (
        "Classify the relevance of the product to the shopping query. "
        "Choose exactly one label: Exact, Substitute, Complement, or Irrelevant."
    )

    product_info = f"Query: {data.query}\n"
    product_info += f"Product Title: {data.product_title}\n"

    if data.brand:
        product_info += f"Brand: {data.brand}\n"

    if data.description:
        product_info += f"Description: {data.description}\n"

    if data.features:
        product_info += f"Features: {data.features}\n"

    prompt = f"<s>[INST] {instruction}\n\n{product_info} [/INST]"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    inputs = inputs.to(model.get_input_embeddings().weight.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    match = re.search(
        r"\b(Exact|Substitute|Complement|Irrelevant)\b",
        answer,
        re.IGNORECASE,
    )

    return match.group(1).capitalize() if match else "INVALID"