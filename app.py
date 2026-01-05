import streamlit as st
import numpy as np
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load("RFC_model.pkl")
    scaler = joblib.load("Sc.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter customer details and this app will predict whether the customer is **likely to churn** or **stay**."
)

# Manual encodings from notebook
GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Female": 0, "Male": 1}
CARDTYPE_MAP = {"DIAMOND": 0, "GOLD": 1, "PLATINUM": 2, "SILVER": 3}

with st.form("churn_form"):

    st.subheader("Customer demographics")
    geography = st.selectbox("Geography", list(GEOGRAPHY_MAP.keys()))
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
    age = st.number_input("Age", min_value=18, max_value=100, value=40)

    st.subheader("Account information")
    creditscore = st.number_input("Credit Score", min_value=350, max_value=900, value=650)
    tenure = st.number_input("Tenure (Years with bank)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Account Balance", min_value=0.0, max_value=250898.09, value=100000.0, step=1000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=199992.48, value=100000.0, step=1000.0)

    st.subheader("Service experience")
    complain = st.selectbox("Has Complaint?", ["No", "Yes"])
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    card_type = st.selectbox("Card Type", list(CARDTYPE_MAP.keys()))
    point_earned = st.number_input("Reward Points Earned", min_value=119, max_value=1000, value=500, step=10)

    submitted = st.form_submit_button("Predict Churn")

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
    # Binary mappings
    has_cr_card_bin = 1 if has_cr_card == "Yes" else 0
    is_active_member_bin = 1 if is_active_member == "Yes" else 0
    complain_bin = 1 if complain == "Yes" else 0

    # Manual label encodings
    geo_enc = GEOGRAPHY_MAP[geography]
    gender_enc = GENDER_MAP[gender]
    cardtype_enc = CARDTYPE_MAP[card_type]

    # Feature order must match training
    data = {
        "CreditScore": [creditscore],
        "Geography": [geo_enc],
        "Gender": [gender_enc],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card_bin],
        "IsActiveMember": [is_active_member_bin],
        "EstimatedSalary": [estimated_salary],
        "Complain": [complain_bin],
        "Satisfaction Score": [satisfaction_score],
        "Card Type": [cardtype_enc],
        "Point Earned": [point_earned],
    }

    df = pd.DataFrame(data)

    # Scale with same Sc.pkl
    df_scaled = scaler.transform(df)

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
        else:
            st.success(f"Customer is **NOT likely to churn** (probability of churn: {prob:.3f}).")

    except Exception as e:
        st.error("Error while predicting. Please check logs / console.")
        st.text(str(e))
