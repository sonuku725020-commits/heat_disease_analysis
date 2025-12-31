# ============================================
# 🚀 FASTAPI BACKEND - ChurnPredict
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
import uvicorn

# ============================================
# Initialize FastAPI App
# ============================================

app = FastAPI(
    title="ChurnPredict API",
    description="Heart Disease Prediction API using Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Load Model & Artifacts
# ============================================

try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Model and artifacts loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

# ============================================
# Pydantic Models (Request/Response)
# ============================================

class PatientData(BaseModel):
    Age: int = Field(..., ge=1, le=120, description="Patient Age")
    Sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    ChestPainType: int = Field(..., ge=0, le=3, description="Chest Pain Type (0-3)")
    BP: int = Field(..., ge=50, le=250, description="Blood Pressure")
    Cholesterol: int = Field(..., ge=100, le=600, description="Cholesterol Level")
    FBSOver120: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120")
    EKGResults: int = Field(..., ge=0, le=2, description="EKG Results (0-2)")
    MaxHR: int = Field(..., ge=50, le=250, description="Maximum Heart Rate")
    ExerciseAngina: int = Field(..., ge=0, le=1, description="Exercise Induced Angina")
    STDepression: float = Field(..., ge=0, le=10, description="ST Depression")
    SlopeOfST: int = Field(..., ge=0, le=2, description="Slope of ST (0-2)")
    NumVesselsFluro: int = Field(..., ge=0, le=4, description="Number of Vessels (Fluoroscopy)")
    Thallium: int = Field(..., ge=0, le=3, description="Thallium Test Result")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 55,
                "Sex": 1,
                "ChestPainType": 2,
                "BP": 140,
                "Cholesterol": 250,
                "FBSOver120": 0,
                "EKGResults": 1,
                "MaxHR": 150,
                "ExerciseAngina": 0,
                "STDepression": 1.5,
                "SlopeOfST": 1,
                "NumVesselsFluro": 0,
                "Thallium": 2
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    confidence: float
    recommendations: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_patients: int

# ============================================
# Helper Functions
# ============================================

def preprocess_input(data: PatientData) -> np.ndarray:
    """Preprocess input data for prediction"""
    
    # Create DataFrame
    input_dict = {
        'Age': data.Age,
        'Sex': data.Sex,
        'Chest pain type': data.ChestPainType,
        'BP': data.BP,
        'Cholesterol': data.Cholesterol,
        'FBS over 120': data.FBSOver120,
        'EKG results': data.EKGResults,
        'Max HR': data.MaxHR,
        'Exercise angina': data.ExerciseAngina,
        'ST depression': data.STDepression,
        'Slope of ST': data.SlopeOfST,
        'Number of vessels fluro': data.NumVesselsFluro,
        'Thallium': data.Thallium
    }
    
    df = pd.DataFrame([input_dict])
    
    # Feature Engineering (same as training)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    df['BP_Category'] = pd.cut(df['BP'], bins=[0, 120, 140, 200], labels=[0, 1, 2]).astype(int)
    df['Chol_Risk'] = (df['Cholesterol'] > 200).astype(int)
    df['HR_Risk'] = (df['Max HR'] < 100).astype(int)
    
    # Ensure correct column order
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # Scale features
    scaled_data = scaler.transform(df)
    
    return scaled_data

def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def get_recommendations(probability: float, data: PatientData) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []
    
    if probability > 0.5:
        recommendations.append("🏥 Schedule an appointment with a cardiologist immediately")
    
    if data.Cholesterol > 200:
        recommendations.append("🥗 Reduce cholesterol intake - follow a heart-healthy diet")
    
    if data.BP > 140:
        recommendations.append("💊 Monitor blood pressure regularly and consult doctor")
    
    if data.MaxHR < 100:
        recommendations.append("🏃 Increase physical activity with doctor's approval")
    
    if data.FBSOver120 == 1:
        recommendations.append("🍬 Control blood sugar levels - consider diabetic screening")
    
    if data.Age > 50:
        recommendations.append("📅 Regular annual health checkups recommended")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Maintain healthy lifestyle and regular checkups")
    
    return recommendations

# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PatientData):
    """
    Predict heart disease risk for a single patient
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess
        processed_data = preprocess_input(data)
        
        # Predict
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Get results
        prediction_label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        risk_level = get_risk_level(probability)
        recommendations = get_recommendations(probability, data)
        confidence = max(probability, 1 - probability)
        
        return PredictionResponse(
            prediction=prediction_label,
            probability=round(probability, 4),
            risk_level=risk_level,
            confidence=round(confidence, 4),
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(data: BatchPredictionRequest):
    """
    Predict heart disease risk for multiple patients
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    predictions = []
    
    for patient in data.patients:
        try:
            processed_data = preprocess_input(patient)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            prediction_label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
            risk_level = get_risk_level(probability)
            recommendations = get_recommendations(probability, patient)
            confidence = max(probability, 1 - probability)
            
            predictions.append(PredictionResponse(
                prediction=prediction_label,
                probability=round(probability, 4),
                risk_level=risk_level,
                confidence=round(confidence, 4),
                recommendations=recommendations
            ))
        except Exception as e:
            predictions.append(PredictionResponse(
                prediction="Error",
                probability=0.0,
                risk_level="Unknown",
                confidence=0.0,
                recommendations=[str(e)]
            ))
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_patients=len(predictions)
    )

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "XGBoost Classifier",
        "features": feature_names if feature_names else [],
        "target": "Heart Disease",
        "version": "1.0.0"
    }

# ============================================
# Debug Endpoints
# ============================================

@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to see all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else [],
                "name": route.name,
                "endpoint": route.endpoint.__name__ if hasattr(route, "endpoint") else "N/A"
            })
    
    return {
        "total_routes": len(routes),
        "routes": sorted(routes, key=lambda x: x["path"]),
        "api_docs_url": "http://localhost:8000/docs",
        "redoc_url": "http://localhost:8000/redoc"
    }

@app.get("/debug/info")
async def debug_info():
    """Complete debug information about the API"""
    return {
        "app_info": {
            "title": app.title,
            "version": app.version,
            "description": app.description,
            "docs_url": app.docs_url,
            "redoc_url": app.redoc_url
        },
        "model_status": {
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "feature_names_loaded": feature_names is not None
        },
        "endpoints": {
            "health_check": [
                "GET /",
                "GET /health"
            ],
            "predictions": [
                "POST /predict",
                "POST /predict/batch"
            ],
            "info": [
                "GET /model/info",
                "GET /debug/routes",
                "GET /debug/info",
                "GET /debug/view"
            ]
        },
        "test_urls": {
            "swagger_ui": "http://localhost:8000/docs",
            "redoc": "http://localhost:8000/redoc",
            "health": "http://localhost:8000/health",
            "routes": "http://localhost:8000/debug/routes"
        }
    }

@app.get("/debug/view")
async def debug_view():
    """HTML view of all endpoints"""
    routes_html = ""
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods) if route.methods else "N/A"
            routes_html += f"""
            <tr>
                <td><code>{route.path}</code></td>
                <td><span class="method">{methods}</span></td>
                <td>{route.name}</td>
            </tr>
            """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ChurnPredict API - Endpoints</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            code {{ background-color: #f1f1f1; padding: 2px 4px; border-radius: 3px; }}
            .method {{ background-color: #2196F3; color: white; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
            .links {{ margin-top: 20px; }}
            .links a {{ margin-right: 20px; text-decoration: none; color: #2196F3; }}
            .status {{ margin: 20px 0; padding: 15px; background-color: {'#d4edda' if model else '#f8d7da'}; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>🚀 ChurnPredict API - All Endpoints</h1>
        
        <div class="status">
            <strong>Model Status:</strong> {'✅ Loaded' if model else '❌ Not Loaded'}
        </div>
        
        <table>
            <tr>
                <th>Path</th>
                <th>Method</th>
                <th>Name</th>
            </tr>
            {routes_html}
        </table>
        
        <div class="links">
            <h3>Quick Links:</h3>
            <a href="/docs" target="_blank">📚 Swagger UI</a>
            <a href="/redoc" target="_blank">📖 ReDoc</a>
            <a href="/debug/routes" target="_blank">🔍 JSON Routes</a>
            <a href="/health" target="_blank">💚 Health Check</a>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/openapi-debug")
async def debug_openapi():
    """Get the OpenAPI schema for debugging"""
    return app.openapi()

@app.get("/routes")
async def get_all_routes():
    """List all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"total_routes": len(routes), "routes": routes}

# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)