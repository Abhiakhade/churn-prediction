from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import pandas as pd
import os
import urllib.request
import shap
from pathlib import Path

# -----------------------------
# App Init
# -----------------------------
app = FastAPI(title="Customer Churn Prediction API")

# Enable CORS (important for UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="src/templates")

# -----------------------------
# Download model if not exists
# -----------------------------
MODEL_URL = "https://your-cloud-storage-link/model.pkl"  # Replace with your URL
MODEL_PATH = Path("models/model.pkl")

def download_model():
    """Download model from cloud storage if not exists"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not MODEL_PATH.exists():
        print(f"Downloading model from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Model downloaded successfully")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise

# Download model before loading
download_model()

# -----------------------------
# Load Model
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]

# Extract pipeline parts
preprocessor = model.named_steps["preprocessing"]
model_step = model.named_steps["model"]

# SHAP explainer (load once)
explainer = shap.TreeExplainer(model_step)

# -----------------------------
# Input Schema
# -----------------------------
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# -----------------------------
# UI Route
# -----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request,                 # ✅ required first
        "index.html",            # template name
        {"request": request}     # context
    )
# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])

        prob = model.predict_proba(df)[0][1]
        prediction = int(prob >= threshold)

        return {
            "churn_probability": round(float(prob), 4),
            "threshold_used": round(float(threshold), 4),
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# SHAP Explain Endpoint
# -----------------------------
@app.post("/explain")
def explain(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])

        X_transformed = preprocessor.transform(df)

        shap_values = explainer.shap_values(X_transformed)

        # Handle XGBoost output format
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_names = preprocessor.get_feature_names_out()

        explanation = dict(zip(feature_names, shap_values[0]))

        # Sort top features
        explanation = dict(
            sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )

        # Convert to JSON-safe
        explanation = {k: float(v) for k, v in explanation.items()}

        return {"top_feature_impacts": explanation}

    except Exception as e:
        return {"error": str(e)}