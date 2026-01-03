from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
import joblib
import numpy as np

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger("HeartPredict")

# Load model and preprocessing
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    logger.info("🚀 Starting HeartPredict AI...")
    logger.info("✅ Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    logger.warning("⚠️ Application started without model")
    model = None
    scaler = None
    feature_names = None

def compute_age_group(age):
    if age < 40:
        return 0
    elif age < 60:
        return 1
    else:
        return 2

def compute_bp_category(bp):
    if bp < 120:
        return 0
    elif bp < 140:
        return 1
    else:
        return 2

def compute_chol_risk(chol):
    return 1 if chol > 200 else 0

def compute_hr_risk(hr):
    return 1 if hr < 120 else 0

class PatientInput(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    BP: int
    Cholesterol: int
    FBSOver120: int
    EKGResults: int
    MaxHR: int
    ExerciseAngina: int
    STDepression: float
    SlopeOfST: int
    NumVesselsFluro: int
    Thallium: int

class PredictionOutput(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    confidence: float
    risk_score: int

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

@app.post("/api/v1/predict", response_model=PredictionOutput)
def predict(data: PatientInput):
    if model is None or scaler is None:
        return PredictionOutput(
            prediction="Error",
            probability=0.0,
            risk_level="Unknown",
            confidence=0.0,
            risk_score=0,
        )

    # Compute additional features
    age_group = compute_age_group(data.Age)
    bp_category = compute_bp_category(data.BP)
    chol_risk = compute_chol_risk(data.Cholesterol)
    hr_risk = compute_hr_risk(data.MaxHR)

    # Create feature vector in the order of feature_names
    features = [
        data.Age,
        data.Sex,
        data.ChestPainType,
        data.BP,
        data.Cholesterol,
        data.FBSOver120,
        data.EKGResults,
        data.MaxHR,
        data.ExerciseAngina,
        data.STDepression,
        data.SlopeOfST,
        data.NumVesselsFluro,
        data.Thallium,
        age_group,
        bp_category,
        chol_risk,
        hr_risk
    ]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict probability
    prob = model.predict_proba(features_scaled)[0][1]

    # Determine prediction
    prediction = "Heart Disease" if prob >= 0.5 else "No Heart Disease"

    # Determine risk level
    if prob < 0.3:
        risk_level = "Low Risk"
    elif prob < 0.7:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    # Confidence and risk score
    confidence = prob
    risk_score = int(prob * 100)

    return PredictionOutput(
        prediction=prediction,
        probability=prob,
        risk_level=risk_level,
        confidence=confidence,
        risk_score=risk_score,
    )
