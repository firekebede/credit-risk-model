import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

from src.api.pydantic_models import RiskPredictionRequest, RiskPredictionResponse

app = FastAPI()

# Load the model from MLflow (adjust the model URI to your actual model path)
MODEL_URI = "models:/credit_risk_classifier@prod"

model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API is running."}

@app.post("/predict", response_model=RiskPredictionResponse)
def predict_risk(data: RiskPredictionRequest):
    # Convert the incoming request to a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    probability = model.predict_proba(input_df)[0][1]

    return RiskPredictionResponse(risk_probability=round(probability, 4))
