from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import load_model
from app.api import predict

app = FastAPI(title="ESCI Product Relevance Classifier")

tokenizer, model = load_model()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(data: PredictionRequest):
    label = predict(data, tokenizer, model)
    return {"label": label}