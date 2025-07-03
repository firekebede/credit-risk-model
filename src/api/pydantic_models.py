
from pydantic import BaseModel

class RiskPredictionRequest(BaseModel):
    # Add all model feature fields used during training here.
    # Example (replace with actual features):
    Recency: float
    Frequency: float
    Monetary: float
    # Add more features if your model uses them

class RiskPredictionResponse(BaseModel):
    risk_probability: float
