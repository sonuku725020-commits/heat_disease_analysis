import streamlit as st
import requests

API_URL = "http://localhost:8000/api/v1/predict"
HEALTH_URL = "http://localhost:8000/health"

st.title("HeartPredict – Minimal Demo")

# health check
try:
    r = requests.get(HEALTH_URL, timeout=2)
    if r.status_code == 200:
        st.success("Backend online")
    else:
        st.error("Backend health check failed")
except Exception as e:
    st.error(f"Backend not reachable: {e}")

st.subheader("Patient input")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 55)
    sex = st.selectbox("Sex", ["Female", "Male"])
    bp = st.number_input("Resting BP", 50, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
with col2:
    max_hr = st.number_input("Max HR", 50, 250, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
    chest_pain = st.slider("Chest Pain Type (0-3)", 0, 3, 0)
    st_dep = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1)

if st.button("Predict"):
    payload = {
        "Age": int(age),
        "Sex": 1 if sex == "Male" else 0,
        "ChestPainType": int(chest_pain),
        "BP": int(bp),
        "Cholesterol": int(chol),
        "FBSOver120": 0,
        "EKGResults": 0,
        "MaxHR": int(max_hr),
        "ExerciseAngina": 1 if exercise_angina == "Yes" else 0,
        "STDepression": float(st_dep),
        "SlopeOfST": 0,
        "NumVesselsFluro": 0,
        "Thallium": 0,
    }

    try:
        res = requests.post(API_URL, json=payload, timeout=5)
        res.raise_for_status()
        data = res.json()
        st.success(
            f"Prediction: {data['prediction']}\n\n"
            f"Risk level: {data['risk_level']}\n\n"
            f"Risk score: {data['risk_score']} (prob={data['probability']:.2f})"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
