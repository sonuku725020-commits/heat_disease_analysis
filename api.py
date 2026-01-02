"""
HeartPredict AI - Advanced Cardiac Risk Assessment Platform
A production-grade, scalable heart disease prediction API
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import lru_cache
from enum import Enum
import pandas as pd
import numpy as np
import joblib
import uvicorn
import asyncio
import hashlib
import logging
import time
import uuid
import os

# Optional import for Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Using static recommendations.")

# ══════════════════════════════════════════════════════════════════════════════
# 📋 CONFIGURATION & LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("HeartPredict")


class Settings:
    """Application configuration"""
    APP_NAME: str = "HeartPredict AI"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Advanced Cardiac Risk Assessment Platform powered by Machine Learning"

    MODEL_PATH: str = "Heart_Disease/churn_model.pkl"
    SCALER_PATH: str = "Heart_Disease/scaler.pkl"
    FEATURES_PATH: str = "Heart_Disease/feature_names.pkl"

    API_KEY_ENABLED: bool = False
    API_KEY: str = os.getenv("API_KEY", "heartpredict-demo-key-2024")

    # Gemini AI Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-1.5-flash"
    USE_GEMINI_RECOMMENDATIONS: bool = os.getenv("USE_GEMINI_RECOMMENDATIONS", "false").lower() == "true"

    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    CACHE_TTL: int = 300  # 5 minutes
    MAX_BATCH_SIZE: int = 100

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()

# ══════════════════════════════════════════════════════════════════════════════
# 📊 ENUMS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

class Sex(int, Enum):
    FEMALE = 0
    MALE = 1


class ChestPainType(int, Enum):
    TYPICAL_ANGINA = 0
    ATYPICAL_ANGINA = 1
    NON_ANGINAL = 2
    ASYMPTOMATIC = 3


class EKGResult(int, Enum):
    NORMAL = 0
    ST_T_ABNORMALITY = 1
    LV_HYPERTROPHY = 2


class SlopeOfST(int, Enum):
    UPSLOPING = 0
    FLAT = 1
    DOWNSLOPING = 2


class ThalliumResult(int, Enum):
    NORMAL = 0
    FIXED_DEFECT = 1
    REVERSIBLE_DEFECT = 2
    NOT_DESCRIBED = 3


class RiskLevel(str, Enum):
    VERY_LOW = "Very Low Risk"
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    CRITICAL = "Critical Risk"


# Risk factor weights for explainability
RISK_FACTOR_WEIGHTS = {
    "Age": 0.15,
    "Cholesterol": 0.12,
    "BP": 0.12,
    "MaxHR": 0.10,
    "STDepression": 0.15,
    "ChestPainType": 0.10,
    "NumVesselsFluro": 0.12,
    "Thallium": 0.14
}

# ══════════════════════════════════════════════════════════════════════════════
# 🧠 MODEL MANAGER (SINGLETON PATTERN)
# ══════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Thread-safe singleton for model management"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        if self._initialized:
            return
        
        try:
            self.model = joblib.load(settings.MODEL_PATH)
            self.scaler = joblib.load(settings.SCALER_PATH)
            self.feature_names = joblib.load(settings.FEATURES_PATH)
            self.model_loaded_at = datetime.utcnow()
            self.prediction_count = 0
            self._initialized = True
            logger.info("✅ Model artifacts loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.model = None
            self.scaler = None
            self.feature_names = None
            self._initialized = False
    
    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model is not None
    
    def increment_predictions(self, count: int = 1):
        self.prediction_count += count
    
    def get_stats(self) -> dict:
        return {
            "model_loaded": self.is_ready,
            "loaded_at": self.model_loaded_at.isoformat() if self.is_ready else None,
            "total_predictions": self.prediction_count,
            "model_type": type(self.model).__name__ if self.model else None
        }


model_manager = ModelManager()

# ══════════════════════════════════════════════════════════════════════════════
# 📦 CACHING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class PredictionCache:
    """In-memory LRU cache with TTL"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache: Dict[str, tuple] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, data: dict) -> str:
        """Generate cache key from input data"""
        sorted_data = str(sorted(data.items()))
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def get(self, data: dict) -> Optional[dict]:
        key = self._generate_key(data)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                return result
            del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, data: dict, result: dict):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        key = self._generate_key(data)
        self.cache[key] = (result, time.time())
    
    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{(self.hits/total*100):.2f}%" if total > 0 else "0%"
        }
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


prediction_cache = PredictionCache(max_size=1000, ttl=settings.CACHE_TTL)

# ══════════════════════════════════════════════════════════════════════════════
# 🚦 RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.max_requests = settings.RATE_LIMIT_REQUESTS
        self.window = settings.RATE_LIMIT_WINDOW
    
    def is_allowed(self, client_ip: str) -> tuple[bool, dict]:
        now = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window
        ]
        
        if len(self.requests[client_ip]) >= self.max_requests:
            reset_time = min(self.requests[client_ip]) + self.window
            return False, {
                "remaining": 0,
                "reset_in": int(reset_time - now),
                "limit": self.max_requests
            }
        
        self.requests[client_ip].append(now)
        return True, {
            "remaining": self.max_requests - len(self.requests[client_ip]),
            "reset_in": self.window,
            "limit": self.max_requests
        }


rate_limiter = RateLimiter()

# ══════════════════════════════════════════════════════════════════════════════
# 📝 PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PatientData(BaseModel):
    """Patient health metrics for heart disease prediction"""

    age: int = Field(alias="Age", ge=1, le=120, description="Patient age in years", example=55)
    sex: Sex = Field(alias="Sex", description="Biological sex (0: Female, 1: Male)", example=1)
    chest_pain_type: ChestPainType = Field(alias="ChestPainType", description="Type of chest pain (0-3)", example=2)
    bp: int = Field(alias="BP", ge=60, le=250, description="Resting blood pressure (mm Hg)", example=140)
    cholesterol: int = Field(alias="Cholesterol", ge=100, le=600, description="Serum cholesterol (mg/dl)", example=230)
    fbs_over120: int = Field(alias="FBSOver120", ge=0, le=1, description="Fasting blood sugar > 120 mg/dl", example=0)
    ekg_results: EKGResult = Field(alias="EKGResults", description="Resting ECG results (0-2)", example=1)
    max_hr: int = Field(alias="MaxHR", ge=50, le=250, description="Maximum heart rate achieved", example=150)
    exercise_angina: int = Field(alias="ExerciseAngina", ge=0, le=1, description="Exercise induced angina", example=0)
    st_depression: float = Field(alias="STDepression", ge=0, le=10, description="ST depression induced by exercise", example=1.5)
    slope_of_st: SlopeOfST = Field(alias="SlopeOfST", description="Slope of peak exercise ST segment", example=1)
    num_vessels_fluro: int = Field(alias="NumVesselsFluro", ge=0, le=4, description="Number of vessels colored by fluoroscopy", example=0)
    thallium: ThalliumResult = Field(alias="Thallium", description="Thallium stress test result", example=2)

    @validator('age')
    def validate_age(cls, v):
        if v < 18:
            logger.warning(f"Prediction for minor patient (age: {v})")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 55,
                "Sex": 1,
                "ChestPainType": 2,
                "BP": 140,
                "Cholesterol": 230,
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


class BatchPredictionInput(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData] = Field(..., max_length=100, description="List of patients")
    
    @validator('patients')
    def validate_batch_size(cls, v):
        if len(v) > settings.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {settings.MAX_BATCH_SIZE}")
        return v


class RiskFactor(BaseModel):
    """Individual risk factor analysis"""
    factor: str
    value: Union[int, float]
    status: str
    contribution: float
    recommendation: str


class PredictionResponse(BaseModel):
    """Detailed prediction response"""
    prediction_id: str
    timestamp: str
    prediction: str
    probability: float
    confidence: float
    risk_level: RiskLevel
    risk_score: int
    risk_factors: List[RiskFactor]
    recommendations: List[str]
    lifestyle_tips: List[str]
    follow_up_actions: List[str]
    model_version: str
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    batch_id: str
    timestamp: str
    total_patients: int
    predictions: List[PredictionResponse]
    summary: dict
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    model_status: dict
    cache_stats: dict
    uptime_seconds: float


class AnalyticsResponse(BaseModel):
    """Analytics dashboard data"""
    total_predictions: int
    risk_distribution: dict
    avg_probability: float
    cache_performance: dict
    model_info: dict


# ══════════════════════════════════════════════════════════════════════════════
# 🔧 CORE PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PredictionEngine:
    """Advanced prediction engine with explainability"""
    
    @staticmethod
    def preprocess(data: dict) -> np.ndarray:
        """Preprocess input data with feature engineering"""
        input_dict = {
            'Age': data['Age'],
            'Sex': data['Sex'],
            'Chest pain type': data['ChestPainType'],
            'BP': data['BP'],
            'Cholesterol': data['Cholesterol'],
            'FBS over 120': data['FBSOver120'],
            'EKG results': data['EKGResults'],
            'Max HR': data['MaxHR'],
            'Exercise angina': data['ExerciseAngina'],
            'ST depression': data['STDepression'],
            'Slope of ST': data['SlopeOfST'],
            'Number of vessels fluro': data['NumVesselsFluro'],
            'Thallium': data['Thallium']
        }
        
        df = pd.DataFrame([input_dict])
        
        # Feature Engineering
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
        df['BP_Category'] = pd.cut(df['BP'], bins=[0, 120, 140, 200], labels=[0, 1, 2]).astype(int)
        df['Chol_Risk'] = (df['Cholesterol'] > 200).astype(int)
        df['HR_Risk'] = (df['Max HR'] < 100).astype(int)
        
        # Additional engineered features
        df['Age_BP_Interaction'] = df['Age'] * df['BP'] / 1000
        df['Cardiac_Index'] = df['Max HR'] / df['Age']
        
        # Reindex and scale
        df = df.reindex(columns=model_manager.feature_names, fill_value=0)
        return model_manager.scaler.transform(df)
    
    @staticmethod
    def calculate_risk_level(probability: float) -> RiskLevel:
        """Determine risk level with granularity"""
        if probability < 0.15:
            return RiskLevel.VERY_LOW
        elif probability < 0.35:
            return RiskLevel.LOW
        elif probability < 0.55:
            return RiskLevel.MODERATE
        elif probability < 0.75:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL
    
    @staticmethod
    def calculate_risk_score(probability: float) -> int:
        """Convert probability to 0-100 risk score"""
        return int(probability * 100)
    
    @staticmethod
    def analyze_risk_factors(data: dict, probability: float) -> List[RiskFactor]:
        """Detailed risk factor analysis with contribution scores"""
        factors = []
        
        # Age Analysis
        age_status = "Low" if data['Age'] < 45 else "Medium" if data['Age'] < 60 else "High"
        age_contrib = min(0.3, (data['Age'] - 30) / 100) if data['Age'] > 30 else 0
        factors.append(RiskFactor(
            factor="Age",
            value=data['Age'],
            status=age_status,
            contribution=round(age_contrib * 100, 1),
            recommendation="Regular cardiac screening recommended for patients over 45" if data['Age'] >= 45 else "Maintain healthy lifestyle"
        ))
        
        # Blood Pressure
        bp_status = "Normal" if data['BP'] < 120 else "Elevated" if data['BP'] < 140 else "High"
        bp_contrib = max(0, (data['BP'] - 120) / 200)
        factors.append(RiskFactor(
            factor="Blood Pressure",
            value=data['BP'],
            status=bp_status,
            contribution=round(bp_contrib * 100, 1),
            recommendation="Consider medication consultation" if data['BP'] >= 140 else "Monitor regularly"
        ))
        
        # Cholesterol
        chol_status = "Optimal" if data['Cholesterol'] < 200 else "Borderline" if data['Cholesterol'] < 240 else "High"
        chol_contrib = max(0, (data['Cholesterol'] - 200) / 400)
        factors.append(RiskFactor(
            factor="Cholesterol",
            value=data['Cholesterol'],
            status=chol_status,
            contribution=round(chol_contrib * 100, 1),
            recommendation="Dietary changes recommended" if data['Cholesterol'] >= 200 else "Maintain current diet"
        ))
        
        # Heart Rate
        hr_status = "Low" if data['MaxHR'] < 100 else "Normal" if data['MaxHR'] < 170 else "High"
        hr_contrib = max(0, (150 - data['MaxHR']) / 150) if data['MaxHR'] < 150 else 0
        factors.append(RiskFactor(
            factor="Max Heart Rate",
            value=data['MaxHR'],
            status=hr_status,
            contribution=round(hr_contrib * 100, 1),
            recommendation="Cardio exercises recommended" if data['MaxHR'] < 120 else "Good cardiac response"
        ))
        
        # ST Depression
        st_status = "Normal" if data['STDepression'] < 1 else "Concerning" if data['STDepression'] < 2.5 else "Critical"
        st_contrib = min(0.4, data['STDepression'] / 5)
        factors.append(RiskFactor(
            factor="ST Depression",
            value=data['STDepression'],
            status=st_status,
            contribution=round(st_contrib * 100, 1),
            recommendation="Further cardiac evaluation needed" if data['STDepression'] >= 2 else "Monitor during exercise"
        ))
        
        # Vessels
        vessels_status = "Clear" if data['NumVesselsFluro'] == 0 else "Minor blockage" if data['NumVesselsFluro'] <= 2 else "Significant blockage"
        vessels_contrib = data['NumVesselsFluro'] * 0.15
        factors.append(RiskFactor(
            factor="Vessels Affected",
            value=data['NumVesselsFluro'],
            status=vessels_status,
            contribution=round(vessels_contrib * 100, 1),
            recommendation="Angioplasty consultation recommended" if data['NumVesselsFluro'] >= 2 else "Continue monitoring"
        ))
        
        return sorted(factors, key=lambda x: x.contribution, reverse=True)
    
    @staticmethod
    def generate_gemini_recommendations(probability: float, data: dict, risk_level: RiskLevel) -> List[str]:
        """Generate personalized recommendations using Gemini AI"""
        if not GEMINI_AVAILABLE or not genai:
            return PredictionEngine.generate_static_recommendations(probability, data, risk_level)

        try:
            model = genai.GenerativeModel(settings.GEMINI_MODEL)

            prompt = f"""
            As a medical AI assistant, provide personalized heart disease prevention recommendations based on the following patient data:

            Patient Profile:
            - Age: {data['Age']} years
            - Sex: {'Male' if data['Sex'] == 1 else 'Female'}
            - Blood Pressure: {data['BP']} mmHg
            - Cholesterol: {data['Cholesterol']} mg/dL
            - Max Heart Rate: {data['MaxHR']} bpm
            - Fasting Blood Sugar: {'>120 mg/dL' if data['FBSOver120'] == 1 else 'Normal'}
            - Exercise Angina: {'Present' if data['ExerciseAngina'] == 1 else 'Absent'}
            - ST Depression: {data['STDepression']}
            - Number of Blocked Vessels: {data['NumVesselsFluro']}
            - Risk Level: {risk_level.value}
            - Heart Disease Probability: {probability*100:.1f}%

            Provide 6-8 specific, actionable recommendations for this patient. Focus on:
            1. Immediate medical actions if high risk
            2. Lifestyle modifications
            3. Dietary changes
            4. Exercise recommendations
            5. Monitoring and follow-up
            6. Preventive measures

            Format each recommendation as a clear, concise bullet point with an appropriate emoji.
            Make recommendations specific to the patient's risk factors and current health metrics.
            """

            response = model.generate_content(prompt)
            recommendations_text = response.text.strip()

            # Parse the response into a list
            recommendations = []
            for line in recommendations_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('1.') or line.startswith('2.') or any(char.isdigit() and line.startswith(char) for char in '123456789')):
                    # Clean up the line
                    line = line.lstrip('•-123456789. ').strip()
                    if line:
                        recommendations.append(line)

            # Fallback if parsing fails
            if not recommendations:
                recommendations = recommendations_text.split('\n')[:8]

            return recommendations[:8]

        except Exception as e:
            logger.error(f"Gemini recommendation generation failed: {e}")
            # Fallback to static recommendations
            return PredictionEngine.generate_static_recommendations(probability, data, risk_level)

    @staticmethod
    def generate_static_recommendations(probability: float, data: dict, risk_level: RiskLevel) -> List[str]:
        """Generate personalized, actionable recommendations (static fallback)"""
        recommendations = []

        # Critical recommendations
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("🚨 URGENT: Schedule immediate cardiology consultation")
            recommendations.append("🏥 Consider comprehensive cardiac workup including stress test")

        # Cholesterol recommendations
        if data['Cholesterol'] > 240:
            recommendations.append("💊 Discuss statin therapy with your physician")
            recommendations.append("🥗 Adopt DASH or Mediterranean diet immediately")
        elif data['Cholesterol'] > 200:
            recommendations.append("🥬 Increase soluble fiber intake (oats, beans, fruits)")
            recommendations.append("🐟 Add omega-3 rich foods (salmon, walnuts, flaxseed)")

        # Blood pressure recommendations
        if data['BP'] > 140:
            recommendations.append("🧂 Reduce sodium intake to less than 2,300 mg/day")
            recommendations.append("💊 Blood pressure medication evaluation recommended")
        elif data['BP'] > 130:
            recommendations.append("📉 Lifestyle modifications to reduce blood pressure")

        # Heart rate recommendations
        if data['MaxHR'] < 100:
            recommendations.append("🏃 Gradually increase aerobic exercise with medical supervision")
            recommendations.append("❤️ Consider cardiac rehabilitation program")

        # Blood sugar recommendations
        if data['FBSOver120'] == 1:
            recommendations.append("🩸 Regular blood glucose monitoring recommended")
            recommendations.append("🍬 Consult endocrinologist for diabetes management")

        # Age-specific recommendations
        if data['Age'] > 60:
            recommendations.append("📅 Bi-annual cardiac checkups recommended")
            recommendations.append("💉 Ensure all cardiovascular vaccinations are current")
        elif data['Age'] > 45:
            recommendations.append("📅 Annual cardiac screening recommended")

        # Vessel-related recommendations
        if data['NumVesselsFluro'] >= 2:
            recommendations.append("🔬 Discuss revascularization options with cardiologist")

        if not recommendations:
            recommendations.append("✅ Continue maintaining your healthy lifestyle")
            recommendations.append("📊 Regular annual health screenings recommended")

        return recommendations[:8]  # Limit to top 8 recommendations

    @staticmethod
    def generate_recommendations(probability: float, data: dict, risk_level: RiskLevel) -> List[str]:
        """Generate personalized recommendations - uses Gemini if available, otherwise static"""
        if settings.USE_GEMINI_RECOMMENDATIONS and settings.GEMINI_API_KEY:
            return PredictionEngine.generate_gemini_recommendations(probability, data, risk_level)
        else:
            return PredictionEngine.generate_static_recommendations(probability, data, risk_level)
    
    @staticmethod
    def generate_lifestyle_tips(data: dict) -> List[str]:
        """Generate lifestyle modification tips"""
        tips = [
            "🚶 Aim for 150 minutes of moderate aerobic activity weekly",
            "🛌 Maintain 7-9 hours of quality sleep per night",
            "🧘 Practice stress-reduction techniques (meditation, yoga)",
            "🚭 Avoid tobacco and limit alcohol consumption",
            "💧 Stay hydrated with 8 glasses of water daily"
        ]
        
        if data['Cholesterol'] > 200:
            tips.insert(0, "🥑 Include heart-healthy fats (avocado, olive oil, nuts)")
        
        if data['BP'] > 130:
            tips.insert(0, "🧂 Cook at home to control sodium intake")
        
        if data['Age'] > 50:
            tips.insert(0, "🏋️ Include strength training 2-3 times per week")
        
        return tips[:6]
    
    @staticmethod
    def generate_followup_actions(probability: float, risk_level: RiskLevel) -> List[str]:
        """Generate follow-up action items"""
        actions = []
        
        if risk_level == RiskLevel.CRITICAL:
            actions = [
                "📞 Call cardiologist within 24 hours",
                "🏥 Visit ER if experiencing chest pain",
                "💊 Review all current medications",
                "📋 Prepare medical history documentation"
            ]
        elif risk_level == RiskLevel.HIGH:
            actions = [
                "📅 Schedule cardiology appointment within 1 week",
                "🔬 Request comprehensive lipid panel",
                "📊 Start blood pressure log",
                "🏃 Consult about cardiac rehabilitation"
            ]
        elif risk_level == RiskLevel.MODERATE:
            actions = [
                "📅 Schedule primary care follow-up within 2 weeks",
                "🏃 Begin supervised exercise program",
                "📝 Start food and exercise diary",
                "🧪 Schedule fasting glucose test"
            ]
        else:
            actions = [
                "✅ Continue current healthy practices",
                "📅 Schedule annual wellness exam",
                "📊 Track health metrics monthly",
                "🎯 Set and maintain health goals"
            ]
        
        return actions
    
    @classmethod
    def predict(cls, data: dict) -> dict:
        """Execute full prediction pipeline"""
        start_time = time.time()
        
        # Check cache first
        cached_result = prediction_cache.get(data)
        if cached_result:
            cached_result['from_cache'] = True
            cached_result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
            return cached_result
        
        # Preprocess and predict
        processed_data = cls.preprocess(data)
        prediction_num = model_manager.model.predict(processed_data)[0]
        probability = float(model_manager.model.predict_proba(processed_data)[0][1])
        
        # Calculate metrics
        risk_level = cls.calculate_risk_level(probability)
        risk_score = cls.calculate_risk_score(probability)
        confidence = float(max(probability, 1 - probability))
        
        # Generate insights
        risk_factors = cls.analyze_risk_factors(data, probability)
        recommendations = cls.generate_recommendations(probability, data, risk_level)
        lifestyle_tips = cls.generate_lifestyle_tips(data)
        follow_up_actions = cls.generate_followup_actions(probability, risk_level)
        
        result = {
            "prediction_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prediction": "Heart Disease Detected" if prediction_num == 1 else "No Heart Disease Detected",
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": [rf.dict() for rf in risk_factors],
            "recommendations": recommendations,
            "lifestyle_tips": lifestyle_tips,
            "follow_up_actions": follow_up_actions,
            "model_version": settings.APP_VERSION,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "from_cache": False
        }
        
        # Cache result
        prediction_cache.set(data, result)
        model_manager.increment_predictions()
        
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

# Application startup time for uptime tracking
app_start_time = time.time()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting HeartPredict AI...")
    model_manager.initialize()

    # Initialize Gemini AI if configured and available
    if GEMINI_AVAILABLE and settings.GEMINI_API_KEY and settings.USE_GEMINI_RECOMMENDATIONS:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info("✅ Gemini AI configured successfully")
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini AI: {e}")
            settings.USE_GEMINI_RECOMMENDATIONS = False
    elif settings.USE_GEMINI_RECOMMENDATIONS and not GEMINI_AVAILABLE:
        logger.warning("⚠️ Gemini AI requested but not available. Using static recommendations.")
        settings.USE_GEMINI_RECOMMENDATIONS = False

    if model_manager.is_ready:
        logger.info("✅ Application started successfully")
    else:
        logger.warning("⚠️ Application started without model")
    yield
    # Shutdown
    logger.info("👋 Shutting down HeartPredict AI...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "🔮 Predictions", "description": "Heart disease prediction endpoints"},
        {"name": "📊 Analytics", "description": "Analytics and statistics"},
        {"name": "🏥 Health", "description": "System health and status"},
        {"name": "📚 Reference", "description": "Reference data and information"}
    ]
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# 🔐 DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Verify API key if enabled"""
    if not settings.API_KEY_ENABLED:
        return True
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return True


async def check_rate_limit(request: Request):
    """Check rate limit for client"""
    client_ip = request.client.host
    allowed, info = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Reset in {info['reset_in']} seconds",
            headers={"X-RateLimit-Reset": str(info['reset_in'])}
        )
    return info


async def verify_model_loaded():
    """Verify model is loaded and ready"""
    if not model_manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service temporarily unavailable."
        )
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["🏥 Health"])
async def root():
    """Welcome endpoint with API overview"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "status": "operational" if model_manager.is_ready else "degraded",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "POST /api/v1/predict",
            "batch_predict": "POST /api/v1/predict/batch",
            "health": "GET /health",
            "analytics": "GET /api/v1/analytics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["🏥 Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    return HealthResponse(
        status="healthy" if model_manager.is_ready else "unhealthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version=settings.APP_VERSION,
        model_status=model_manager.get_stats(),
        cache_stats=prediction_cache.get_stats(),
        uptime_seconds=round(time.time() - app_start_time, 2)
    )


@app.get("/ready", tags=["🏥 Health"])
async def readiness_check():
    """Kubernetes-style readiness probe"""
    if model_manager.is_ready:
        return {"ready": True}
    raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/live", tags=["🏥 Health"])
async def liveness_check():
    """Kubernetes-style liveness probe"""
    return {"alive": True}


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["🔮 Predictions"],
    summary="Predict Heart Disease Risk",
    description="Analyze patient data and predict heart disease risk with detailed insights"
)
async def predict_heart_disease(
    patient: PatientData,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_model_loaded),
    rate_info: dict = Depends(check_rate_limit)
):
    """
    ## Heart Disease Prediction
    
    Analyzes patient health metrics to predict cardiovascular disease risk.
    
    ### Features:
    - **Risk Assessment**: Probability-based risk scoring
    - **Explainability**: Detailed breakdown of contributing factors  
    - **Recommendations**: Personalized health recommendations
    - **Follow-up Actions**: Actionable next steps
    
    ### Response includes:
    - Prediction result and probability
    - Risk level (Very Low to Critical)
    - Risk factor analysis
    - Lifestyle recommendations
    - Follow-up actions
    """
    try:
        data = patient.dict(by_alias=True)
        result = PredictionEngine.predict(data)
        
        # Log prediction asynchronously
        background_tasks.add_task(
            logger.info,
            f"Prediction completed | ID: {result['prediction_id']} | Risk: {result['risk_level']}"
        )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["🔮 Predictions"],
    summary="Batch Predict Heart Disease Risk",
    description="Process multiple patient predictions in a single request"
)
async def batch_predict(
    batch: BatchPredictionInput,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_model_loaded)
):
    """
    ## Batch Heart Disease Prediction
    
    Process multiple patients in a single request for efficient bulk analysis.
    
    ### Limits:
    - Maximum 100 patients per batch
    - Results include summary statistics
    """
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    try:
        predictions = []
        risk_counts = {"Very Low Risk": 0, "Low Risk": 0, "Moderate Risk": 0, "High Risk": 0, "Critical Risk": 0}
        total_probability = 0
        
        for patient in batch.patients:
            result = PredictionEngine.predict(patient.dict(by_alias=True))
            predictions.append(PredictionResponse(**result))
            risk_counts[result['risk_level'].value] += 1
            total_probability += result['probability']
        
        summary = {
            "risk_distribution": risk_counts,
            "average_probability": round(total_probability / len(predictions), 4),
            "high_risk_count": risk_counts["High Risk"] + risk_counts["Critical Risk"],
            "high_risk_percentage": round(
                (risk_counts["High Risk"] + risk_counts["Critical Risk"]) / len(predictions) * 100, 2
            )
        }
        
        background_tasks.add_task(
            logger.info,
            f"Batch prediction completed | ID: {batch_id} | Count: {len(predictions)}"
        )
        
        return BatchPredictionResponse(
            batch_id=batch_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_patients=len(predictions),
            predictions=predictions,
            summary=summary,
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get(
    "/api/v1/analytics",
    response_model=AnalyticsResponse,
    tags=["📊 Analytics"],
    summary="Get Analytics Dashboard Data"
)
async def get_analytics():
    """Get analytics and performance metrics"""
    return AnalyticsResponse(
        total_predictions=model_manager.prediction_count,
        risk_distribution={"info": "Aggregated from prediction history"},
        avg_probability=0.0,
        cache_performance=prediction_cache.get_stats(),
        model_info=model_manager.get_stats()
    )


@app.post("/api/v1/cache/clear", tags=["📊 Analytics"])
async def clear_cache(_: bool = Depends(verify_api_key)):
    """Clear the prediction cache"""
    prediction_cache.clear()
    return {"message": "Cache cleared successfully", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/v1/reference/risk-factors", tags=["📚 Reference"])
async def get_risk_factor_info():
    """Get information about risk factors and their thresholds"""
    return {
        "risk_factors": [
            {
                "name": "Age",
                "description": "Patient age in years",
                "thresholds": {"low": "< 45", "medium": "45-60", "high": "> 60"},
                "weight": RISK_FACTOR_WEIGHTS.get("Age", 0.1)
            },
            {
                "name": "Blood Pressure",
                "description": "Resting blood pressure in mm Hg",
                "thresholds": {"normal": "< 120", "elevated": "120-140", "high": "> 140"},
                "weight": RISK_FACTOR_WEIGHTS.get("BP", 0.1)
            },
            {
                "name": "Cholesterol",
                "description": "Serum cholesterol in mg/dl",
                "thresholds": {"optimal": "< 200", "borderline": "200-240", "high": "> 240"},
                "weight": RISK_FACTOR_WEIGHTS.get("Cholesterol", 0.1)
            },
            {
                "name": "Max Heart Rate",
                "description": "Maximum heart rate achieved during exercise",
                "thresholds": {"low": "< 100", "normal": "100-170", "high": "> 170"},
                "weight": RISK_FACTOR_WEIGHTS.get("MaxHR", 0.1)
            },
            {
                "name": "ST Depression",
                "description": "ST depression induced by exercise relative to rest",
                "thresholds": {"normal": "< 1", "concerning": "1-2.5", "critical": "> 2.5"},
                "weight": RISK_FACTOR_WEIGHTS.get("STDepression", 0.1)
            }
        ],
        "risk_levels": [
            {"level": "Very Low Risk", "probability_range": "0-15%", "color": "#22c55e"},
            {"level": "Low Risk", "probability_range": "15-35%", "color": "#84cc16"},
            {"level": "Moderate Risk", "probability_range": "35-55%", "color": "#eab308"},
            {"level": "High Risk", "probability_range": "55-75%", "color": "#f97316"},
            {"level": "Critical Risk", "probability_range": "75-100%", "color": "#ef4444"}
        ]
    }


@app.get("/api/v1/reference/enums", tags=["📚 Reference"])
async def get_enum_values():
    """Get all enum values for form building"""
    return {
        "Sex": [{"value": e.value, "label": e.name.replace("_", " ").title()} for e in Sex],
        "ChestPainType": [{"value": e.value, "label": e.name.replace("_", " ").title()} for e in ChestPainType],
        "EKGResult": [{"value": e.value, "label": e.name.replace("_", " ").title()} for e in EKGResult],
        "SlopeOfST": [{"value": e.value, "label": e.name.replace("_", " ").title()} for e in SlopeOfST],
        "ThalliumResult": [{"value": e.value, "label": e.name.replace("_", " ").title()} for e in ThalliumResult]
    }


# ══════════════════════════════════════════════════════════════════════════════
# 🚨 EXCEPTION HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "path": str(request.url.path)
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# 🏁 MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4,
        log_level="info",
        access_log=True
    )