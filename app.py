import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown("### 🩺 AI-Powered Heart Risk Assessment")

st.warning("⚠️ This is an AI tool and NOT a medical diagnosis.")

# -----------------------------
# Inputs
# -----------------------------
age = st.slider("Age", 20, 100, 50)

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)

exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)

slope = st.selectbox("Slope of Peak Exercise", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (CA)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [1, 2, 3])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Correct column order
    input_df = input_df[[
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal"
    ]]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # -----------------------------
    # 🔥 Prediction + Sigmoid Calibration
    # -----------------------------
    raw_prob = model.predict_proba(input_scaled)[0][1]

    # Smooth probability
    prob = 1 / (1 + np.exp(-3 * (raw_prob - 0.5)))

    # -----------------------------
    # Result
    # -----------------------------
    st.subheader("🧾 Result")

    if prob < 0.7:
        st.success(f"💚 Low Risk ({prob*100:.2f}%)")
    elif prob < 0.85:
        st.warning(f"🟡 Moderate Risk ({prob*100:.2f}%)")
    else:
        st.error(f"💔 High Risk ({prob*100:.2f}%)")

    st.progress(int(prob * 100))
    st.info(f"📊 Risk Score: {prob*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption("Built with Streamlit • For educational use only")
