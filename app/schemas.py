from pydantic import BaseModel


class PredictionRequest(BaseModel):
    query: str
    product_title: str
    brand: str | None = None
    description: str | None = None
    features: str | None = None


class PredictionResponse(BaseModel):
    label: str