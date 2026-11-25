
# E-Commerce Search Relevance Classification with LLMs (QLoRA)

An end-to-end LLM engineering project that fine-tunes **Mistral-7B-Instruct-v0.2** using QLoRA to classify e-commerce search query–product relevance into Amazon's four ESCI categories: Exact, Substitute, Complement, and Irrelevant.

The project covers data processing, parameter-efficient fine-tuning, model evaluation, inference, and REST API deployment.

## Key Highlights

- **QLoRA fine-tuning:** 4-bit NF4 quantization with LoRA adapters (rank 16).
- **Dataset:** Amazon Shopping Queries (ESCI), using U.S. query–product examples.
- **Data engineering:** Query-level train/validation splitting to prevent data leakage; balanced training data with naturally distributed validation and test sets.
- **Model evaluation:** Compared the base model and fine-tuned model using accuracy, macro-F1, precision, and recall.
- **Inference:** 4-bit Mistral-7B with the trained LoRA adapter, loaded once and reused for predictions.
- **API deployment:** FastAPI REST service with health-check and prediction endpoints.
- **Containerization:** Docker configuration for reproducible application packaging.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · PEFT · BitsAndBytes · FastAPI · Pydantic · Docker · Google Colab GPU

## Model & Training

| Configuration | Value |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning | QLoRA / PEFT |
| Quantization | 4-bit NF4 + double quantization |
| LoRA rank / alpha | 16 / 32 |
| Training examples | 10,000 balanced examples |
| Epochs | 1 |
| Effective batch size | 16 |
| Optimizer steps | 625 |
| Learning rate | 1e-4 |
| Training GPU | Google Colab T4 |

## Evaluation Results

Both models were evaluated on the same examples using deterministic generation.

**Balanced validation sample — 2,000 examples**

| Metric | Base Model | Base + LoRA |
|---|---:|---:|
| Accuracy | 29.75% | **66.40%** |
| Macro-F1 | 31.79% | **66.23%** |
| Invalid predictions | 786 | **0** |

**Naturally distributed test sample — 2,000 examples**

| Metric | Base Model | Base + LoRA |
|---|---:|---:|
| Accuracy | 12.20% | **65.00%** |
| Macro-F1 | 20.17% | **51.48%** |
| Invalid predictions | 1,074 | **0** |

*Evaluation note: The reported metrics use 2,000-example samples, not the complete validation or test datasets. The test sample preserves the natural class distribution. Full test-set evaluation remains future work.*

## Inference & API Deployment

The fine-tuned model is served through a modular FastAPI application.

| Endpoint | Description |
|---|---|
| `GET /health` | Checks API availability |
| `POST /predict` | Accepts a shopping query and product details; returns an ESCI label |

**Deployment architecture:**

Client → FastAPI → Mistral-7B + LoRA → ESCI prediction

The application was designed for GPU-based inference. Google Colab GPU and Cloudflare Tunnel provide a temporary environment for API testing and demonstration. Docker configuration supports reproducible packaging for future cloud deployment.

## Project Structure

```text
Fine_Tuning_LoRA/
├── app/                    # FastAPI application
├── configs/                # Model configuration
├── data/
│   ├── raw/esci/           # Original ESCI dataset
│   └── processed/          # Processed datasets
├── src/
│   ├── data_processing/    # Data preparation
│   ├── training_evaluation/# Fine-tuning and evaluation
│   └── inference/          # Model inference
├── Dockerfile
├── .dockerignore
├── lora-adapter.zip
└── README.md
```

## Future Improvements

- Evaluate the complete, naturally distributed test set.
- Improve minority-class precision, particularly for Complement and Irrelevant.
- Add automated API tests, latency benchmarks, and monitoring.
- Deploy the containerized application on persistent GPU infrastructure.

**Outcome:** A practical LLM classification pipeline demonstrating parameter-efficient fine-tuning, evaluation, model serving, API development, and deployment-oriented engineering.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · PEFT · BitsAndBytes · scikit-learn · Google Colab
