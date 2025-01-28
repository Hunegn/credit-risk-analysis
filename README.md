from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API to serve the trained machine-learning models for predicting credit risk.",
    version="1.0.0"
)

# Load the trained model
model_path = "../models/best_model.pkl"
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

# Define input data schema
class PredictionInput(BaseModel):
    TransactionId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    TransactionStartTime: str  # ISO format
    PricingStrategy: int

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Convert input to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Preprocess the input data (if required)
        input_data['TransactionStartTime'] = pd.to_datetime(input_data['TransactionStartTime'])
        input_data['Weekday'] = input_data['TransactionStartTime'].dt.weekday
        input_data['IsWeekend'] = input_data['Weekday'].isin([5, 6]).astype(int)
        input_data = input_data.drop(columns=['TransactionId', 'TransactionStartTime', 'AccountId', 'SubscriptionId'])

        # Predict using the loaded model
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        return {
            "prediction": int(prediction[0]),
            "risk_probability": float(prediction_proba[0]),
            "message": "Prediction made successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the request: {e}")

# Define a health check endpoint
@app.get("/")
def read_root():
    return {"message": "API is running. Use /predict endpoint for predictions."}
