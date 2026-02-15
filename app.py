import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")
st.caption("Hybrid ML + Medical Rule Based Decision System")

# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    gender_encoder = pickle.load(open("gender_encoder.pkl", "rb"))
    feature_order = pickle.load(open("feature_order.pkl", "rb"))
    return model, gender_encoder, feature_order

model, gender_encoder, feature_order = load_artifacts()

# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.header("Patient Details")

gender = st.sidebar.selectbox("Gender", ["F", "M"])
age = st.sidebar.number_input("Age", 1, 120, 45)

urea = st.sidebar.number_input("Urea (mg/dL)", 10.0, 80.0, 30.0)
cr = st.sidebar.number_input("Creatinine (mg/dL)", 0.4, 3.0, 1.0)
hba1c = st.sidebar.number_input("HbA1c (%)", 4.0, 14.0, 6.0)

chol = st.sidebar.number_input("Cholesterol (mg/dL)", 100.0, 350.0, 190.0)
tg = st.sidebar.number_input("Triglycerides (mg/dL)", 50.0, 400.0, 150.0)
hdl = st.sidebar.number_input("HDL (mg/dL)", 20.0, 90.0, 45.0)
ldl = st.sidebar.number_input("LDL (mg/dL)", 50.0, 250.0, 110.0)
vldl = st.sidebar.number_input("VLDL (mg/dL)", 10.0, 80.0, 30.0)

bmi = st.sidebar.number_input("BMI", 15.0, 50.0, 25.0)

# -------------------------------
# Predict
# -------------------------------
if st.button("🔍 Predict Diabetes Status"):

    # Encode gender
    gender_encoded = gender_encoder.transform([gender])[0]

    # Build input dataframe (SAFE)
    input_df = pd.DataFrame([{
        "Gender": gender_encoded,
        "AGE": age,
        "Urea": urea,
        "Cr": cr,
        "HbA1c": hba1c,
        "Chol": chol,
        "TG": tg,
        "HDL": hdl,
        "LDL": ldl,
        "VLDL": vldl,
        "BMI": bmi
    }])

    # Force correct feature order
    input_df = input_df[feature_order]

    st.write("📊 Input data:", input_df)

    # ML prediction
    ml_prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    prob_dict = {
        cls: float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }

    # -------------------------------
    # MEDICAL RULE OVERRIDE (CRITICAL)
    # -------------------------------
    if hba1c < 5.7:
        final_label = "Non-Diabetic"
        reason = "HbA1c < 5.7 (Normal range)"
    elif 5.7 <= hba1c < 6.5:
        final_label = "Pre-Diabetic"
        reason = "HbA1c between 5.7 and 6.4"
    else:
        final_label = "Diabetic"
        reason = "HbA1c ≥ 6.5"

    # -------------------------------
    # Output
    # -------------------------------
    st.success(f"Prediction: {final_label}")
    st.info(f"Reason: {reason}")
    st.info(f"Confidence (ML Risk Score): {max(prob_dict.values()) * 100:.2f}%")

    st.write("📈 Model Risk Probabilities:", prob_dict)