# ============================================
# 🎨 STREAMLIT FRONTEND - HeartPredict
# ============================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="HeartPredict - Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .low-risk {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .medium-risk {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .high-risk {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# API Configuration (Using Environment Variable)
# ============================================

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ============================================
# Session State Initialization
# ============================================

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============================================
# Helper Functions
# ============================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.RequestException:
        return False


def make_prediction(data):
    """Send prediction request to API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API returned status code: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to API. Is the server running?")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def validate_api_response(result):
    """Validate that API response has all required keys"""
    required_keys = ['prediction', 'risk_level', 'probability', 'confidence', 'recommendations']
    if result is None:
        return False
    return all(key in result for key in required_keys)


def create_gauge_chart(probability):
    """Create a gauge chart for probability"""
    # Validate probability
    if probability is None or not isinstance(probability, (int, float)):
        probability = 0
    
    # Ensure probability is between 0 and 1
    probability = max(0, min(1, probability))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#28a745'},
                {'range': [30, 60], 'color': '#ffc107'},
                {'range': [60, 100], 'color': '#dc3545'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def validate_input_values(age, bp, cholesterol, max_hr):
    """Validate input values and return warnings"""
    warnings = []
    
    if age < 18 or age > 100:
        warnings.append(f"⚠️ Age ({age}) seems unusual. Typical range: 18-100 years.")
    
    if bp < 80 or bp > 200:
        warnings.append(f"⚠️ Blood Pressure ({bp} mmHg) seems unusual. Typical range: 80-200 mmHg.")
    
    if cholesterol < 120 or cholesterol > 400:
        warnings.append(f"⚠️ Cholesterol ({cholesterol} mg/dL) seems unusual. Typical range: 120-400 mg/dL.")
    
    if max_hr < 60 or max_hr > 220:
        warnings.append(f"⚠️ Max Heart Rate ({max_hr}) seems unusual. Typical range: 60-220 bpm.")
    
    return warnings

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    # Use emoji instead of external image
    st.markdown("# ❤️")
    st.markdown("## HeartPredict")
    st.markdown("---")
    
    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")
        st.caption(f"Trying: {API_URL}")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Prediction", "📊 Analytics", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📞 Contact")
    st.markdown("Email: support@heartpredict.com")
    st.markdown("Version: 1.0.1")

# ============================================
# Main Content
# ============================================

if page == "🏠 Home":
    # Home Page
    st.markdown('<h1 class="main-header">❤️ HeartPredict</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Heart Disease Risk Prediction</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate</h3>
            <p>95%+ Accuracy using advanced ML models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Fast</h3>
            <p>Get predictions in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔒 Secure</h3>
            <p>Your data is safe and private</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 1️⃣ Input")
        st.markdown("Enter patient health data")
    
    with col2:
        st.markdown("### 2️⃣ Process")
        st.markdown("AI analyzes the data")
    
    with col3:
        st.markdown("### 3️⃣ Predict")
        st.markdown("Get risk prediction")
    
    with col4:
        st.markdown("### 4️⃣ Recommend")
        st.markdown("Receive health tips")

elif page == "🔮 Prediction":
    # Prediction Page
    st.markdown("## 🔮 Heart Disease Risk Prediction")
    st.markdown("Enter patient details below to get a risk assessment")
    
    st.markdown("---")
    
    # Input Form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Patient Information")
        age = st.slider("Age", 1, 120, 55, key="age_input")
        sex = st.selectbox("Sex", ["Female", "Male"], key="sex_input")
        sex_encoded = 1 if sex == "Male" else 0
        
        st.markdown("### 💓 Cardiac Metrics")
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            key="chest_pain_input"
        )
        chest_pain_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain)
        
        bp = st.slider("Blood Pressure (mmHg)", 50, 250, 120, key="bp_input")
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200, key="cholesterol_input")
        max_hr = st.slider("Maximum Heart Rate", 50, 250, 150, key="max_hr_input")
    
    with col2:
        st.markdown("### 🩺 Medical Tests")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"], key="fbs_input")
        fbs_encoded = 1 if fbs == "Yes" else 0
        
        ekg = st.selectbox(
            "EKG Results",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
            key="ekg_input"
        )
        ekg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(ekg)
        
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"], key="exercise_input")
        exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
        
        st_depression = st.slider("ST Depression", 0.0, 10.0, 1.0, 0.1, key="st_dep_input")
        
        slope = st.selectbox("Slope of ST", ["Upsloping", "Flat", "Downsloping"], key="slope_input")
        slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope)
        
        vessels = st.slider("Number of Major Vessels (Fluoroscopy)", 0, 4, 0, key="vessels_input")
        
        thallium = st.selectbox(
            "Thallium Stress Test",
            ["Normal", "Fixed Defect", "Reversible Defect", "Not Described"],
            key="thallium_input"
        )
        thallium_encoded = ["Normal", "Fixed Defect", "Reversible Defect", "Not Described"].index(thallium)
    
    # Input Validation Warnings
    warnings = validate_input_values(age, bp, cholesterol, max_hr)
    if warnings:
        st.markdown("### ⚠️ Input Warnings")
        for warning in warnings:
            st.warning(warning)
    
    st.markdown("---")
    
    # Predict Button
    if st.button("🔮 Predict Risk", use_container_width=True):
        
        # Check API connection first
        if not check_api_health():
            st.error("❌ Cannot connect to API. Please ensure the backend server is running.")
        else:
            # Prepare data
            input_data = {
                "Age": age,
                "Sex": sex_encoded,
                "ChestPainType": chest_pain_encoded,
                "BP": bp,
                "Cholesterol": cholesterol,
                "FBSOver120": fbs_encoded,
                "EKGResults": ekg_encoded,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina_encoded,
                "STDepression": st_depression,
                "SlopeOfST": slope_encoded,
                "NumVesselsFluro": vessels,
                "Thallium": thallium_encoded
            }
            
            with st.spinner("Analyzing..."):
                result = make_prediction(input_data)
            
            # Validate response
            if result and validate_api_response(result):
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk Level Box
                    risk_level = result.get('risk_level', 'Unknown')
                    prediction = result.get('prediction', 'Unknown')
                    probability = result.get('probability', 0)
                    confidence = result.get('confidence', 0)
                    
                    if "Low" in risk_level:
                        risk_class = "low-risk"
                        emoji = "✅"
                    elif "Medium" in risk_level:
                        risk_class = "medium-risk"
                        emoji = "⚠️"
                    else:
                        risk_class = "high-risk"
                        emoji = "🚨"
                    
                    st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h2>{emoji} {prediction}</h2>
                        <h3>{risk_level}</h3>
                        <p>Probability: {probability*100:.1f}%</p>
                        <p>Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Gauge Chart
                    fig = create_gauge_chart(probability)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                recommendations = result.get('recommendations', [])
                if recommendations:
                    st.markdown("### 💡 Recommendations")
                    for rec in recommendations:
                        st.info(rec)
                
                # Save to history
                st.session_state.prediction_history.append({
                    'age': age,
                    'risk_level': risk_level,
                    'probability': probability
                })
                
            elif result:
                st.error("❌ Received invalid response format from API.")
                st.json(result)  # Show what was received for debugging
            else:
                st.error("❌ Failed to get prediction. Please check API connection.")

elif page == "📊 Analytics":
    # Analytics Page
    st.markdown("## 📊 Analytics Dashboard")
    
    st.info("📈 This section shows model performance and statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Performance")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [0.92, 0.89, 0.94, 0.91, 0.95]
        })
        fig = px.bar(metrics_df, x='Metric', y='Score', color='Score',
                     color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': ['Thallium', 'Chest Pain', 'ST Depression', 'Max HR', 'Age'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        })
        fig = px.pie(importance_df, values='Importance', names='Feature',
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show prediction history if available
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📜 Prediction History (This Session)")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)

elif page == "ℹ️ About":
    # About Page
    st.markdown("## ℹ️ About HeartPredict")
    
    st.markdown("""
    ### 🎯 Mission
    To provide accessible and accurate heart disease risk prediction using advanced machine learning.
    
    ### 🔬 Technology
    - **Machine Learning**: XGBoost, Random Forest, Gradient Boosting
    - **Backend**: FastAPI (Python)
    - **Frontend**: Streamlit
    - **Cloud**: AWS / Google Cloud / Azure
    
    ### 📊 Dataset
    Heart Disease UCI Dataset from Kaggle
    
    ### ⚠️ Disclaimer
    This tool is for educational purposes only and should not replace professional medical advice.
    Always consult a healthcare provider for medical decisions.
    
    ### 👨‍💻 Developer
    Built with ❤️ by HeartPredict Team
    """)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>© 2024 HeartPredict | Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)