import os

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

LORA_PATH = os.getenv("LORA_PATH", "/models/lora-adapter")

MAX_NEW_TOKENS = 8