import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")

st.markdown("Enter customer details:")

# -----------------------------
# Inputs
# -----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure", 0, 72)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)

# -----------------------------
# Prepare Data
# -----------------------------
data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Churn"):

    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    result = response.json()

    st.subheader("Prediction Result")

    if "error" in result:
        st.error(result["error"])
    else:
        st.write(f"📉 Churn Probability: {result['churn_probability']}")
        st.write(f"⚖️ Threshold Used: {result['threshold_used']}")

        if result["prediction"] == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

# -----------------------------
# Explain Button
# -----------------------------
if st.button("Explain Prediction"):

    response = requests.post("http://127.0.0.1:8000/explain", json=data)
    result = response.json()

    st.subheader("🔍 Top Feature Impacts")

    if "error" in result:
        st.error(result["error"])
    else:
        st.bar_chart(result["top_feature_impacts"])