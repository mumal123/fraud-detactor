import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('logistic.pkl')


st.title("Customer Fraud Detection App")

st.write("Fill the customer details to check if they are fraudulent.")


gender = st.selectbox("Gender", ["Male", "Female","Others"])
age = st.slider("Age", 18, 80, 30)
country = st.selectbox("Country", ["India", "USA", "UK", "Germany", "Canada","Japan"])  
preferred_category = st.selectbox("Preferred Category", ["Clothing", "Electronics", "Books", "Home", "Beauty"])  # adjust
avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=200.0)
purchase_frequency = st.number_input("Purchase Frequency", min_value=0.0, value=1.0)
email_open_rate = st.slider("Email Open Rate (%)", 0, 100, 50)


input_df = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "country": [country],
    "preferred_category": [preferred_category],
    "avg_order_value": [avg_order_value],
    "purchase_frequency": [purchase_frequency],
    "email_open_rate": [email_open_rate]
})

# One-hot encode to match training format
input_encoded = pd.get_dummies(input_df)
model_features = model.feature_names_in_

# Ensure all expected columns are present
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[model_features]

# Predict
if st.button("Predict Fraud"):
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is likely FRAUDULENT with probability {proba:.2f}")
    else:
        st.success(f"✅ The customer is NOT fraudulent. Probability of fraud: {proba:.2f}")
