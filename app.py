import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Load saved objects
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("RFC_model.pkl")
    scaler = joblib.load("Sc.pkl")
    ml_encoder = joblib.load("multi_label_encoder.pkl")
    return model, scaler, ml_encoder

model, scaler, ml_encoder = load_artifacts()

st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter customer details and this app will predict whether the customer is **likely to churn** or **stay**."
)

# -----------------------------
# Input form
# -----------------------------
with st.form("churn_form"):

    st.subheader("Customer demographics")
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)

    st.subheader("Account information")
    creditscore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    tenure = st.number_input("Tenure (Years with bank)", min_value=0, max_value=50, value=5)
    balance = st.number_input("Account Balance", min_value=0.0, value=100000.0, step=1000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)

    st.subheader("Service experience")
    complain = st.selectbox("Has Complaint?", ["No", "Yes"])
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
    point_earned = st.number_input("Reward Points Earned", min_value=0, value=500, step=10)

    submitted = st.form_submit_button("Predict Churn")

# -----------------------------
# Preprocess & predict
# -----------------------------
def preprocess_input(
    geography,
    gender,
    age,
    creditscore,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary,
    complain,
    satisfaction_score,
    card_type,
    point_earned,
):
    """
    Build a single-row DataFrame with the same columns & order
    used in Banking_churn_Analysis.ipynb before scaling/encoding.
    """
    # Map Yes/No to 1/0
    has_cr_card_bin = 1 if has_cr_card == "Yes" else 0
    is_active_member_bin = 1 if is_active_member == "Yes" else 0
    complain_bin = 1 if complain == "Yes" else 0

    # Original modeling columns (excluding RowNumber, CustomerId, Surname, Exited)
    data = {
        "CreditScore": [creditscore],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card_bin],
        "IsActiveMember": [is_active_member_bin],
        "EstimatedSalary": [estimated_salary],
        "Complain": [complain_bin],
        "Satisfaction Score": [satisfaction_score],
        "Card Type": [card_type],
        "Point Earned": [point_earned],
    }

    df = pd.DataFrame(data)

    # Apply the same multi-label encoder you fitted in notebook
    # ml_encoder is assumed to have a transform method that takes df and returns encoded df
    df_encoded = ml_encoder.transform(df)

    # Scale numeric features with the same scaler
    df_scaled = scaler.transform(df_encoded)

    return df_scaled

if submitted:
    try:
        X_input = preprocess_input(
            geography,
            gender,
            age,
            creditscore,
            tenure,
            balance,
            num_of_products,
            has_cr_card,
            is_active_member,
            estimated_salary,
            complain,
            satisfaction_score,
            card_type,
            point_earned,
        )

        prob = model.predict_proba(X_input)[0, 1]
        pred = model.predict(X_input)[0]

        st.markdown("---")
        st.subheader("Prediction")

        if pred == 1:
            st.error(f"Customer is **LIKELY to churn** (probability: {prob:.3f}).")
            st.write(
                "- Consider retention strategies like personalized offers, fee waivers, or proactive support."
            )
        else:
            st.success(f"Customer is **NOT likely to churn** (probability of churn: {prob:.3f}).")
            st.write(
                "- Maintain engagement with loyalty programs and targeted cross-sell offers."
            )

    except Exception as e:
        st.error("Error while predicting. Please check logs / console.")
        st.text(str(e))
