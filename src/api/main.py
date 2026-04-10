from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import logging
from pathlib import Path
import shap

# -----------------------------
# App Init
# -----------------------------
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# CORS (restrict in production)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="src/templates")

# -----------------------------
# Load Model (NO download)
# -----------------------------
MODEL_PATH = Path("models/model.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]

preprocessor = model.named_steps.get("preprocessing")
model_step = model.named_steps.get("model")

if preprocessor is None or model_step is None:
    raise ValueError("Invalid pipeline structure")

# SHAP (optional heavy component)
explainer = shap.TreeExplainer(model_step)

MODEL_VERSION = "v1.0"

# -----------------------------
# Input Schema
# -----------------------------
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0, le=100)
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
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


# Batch request
class BatchCustomerData(BaseModel):
    customers: List[CustomerData]


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


# -----------------------------
# UI Route
# -----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
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

        logger.info(f"Prediction made: {prob}")

        return {
            "churn_probability": round(float(prob), 4),
            "prediction": prediction,
            "risk_level": (
                "high" if prob > 0.7 else
                "medium" if prob > 0.4 else
                "low"
            ),
            "model_version": MODEL_VERSION
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# -----------------------------
# Batch Prediction (IMPORTANT)
# -----------------------------
@app.post("/batch_predict")
def batch_predict(data: BatchCustomerData):
    try:
        df = pd.DataFrame([c.dict() for c in data.customers])

        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)

        results = []
        for i in range(len(probs)):
            results.append({
                "churn_probability": float(probs[i]),
                "prediction": int(preds[i])
            })

        return {"results": results}

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# -----------------------------
# SHAP Explain Endpoint
# -----------------------------
@app.post("/explain")
def explain(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])
        X_transformed = preprocessor.transform(df)

        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_names = preprocessor.get_feature_names_out()

        explanation = dict(zip(feature_names, shap_values[0]))

        explanation = dict(
            sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )

        explanation = {k: float(v) for k, v in explanation.items()}

        return {"top_feature_impacts": explanation}

    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")