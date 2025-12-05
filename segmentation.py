import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load("kmeans_customer_segmentation_model.pkl")
scaler = joblib.load("scaler_customer_segmentation.pkl")
st.title("Customer Segmentation Prediction")
st.write("Enter customer details to predict the segment.")

age = st.number_input("Age (18-100)", min_value=18, max_value=100, value=30)
income = st.number_input("Income (0-200000)", min_value=0.0, max_value=200000.0, value=50000.0)
total_spending = st.number_input("Total Spending(sum of all purchases) (0-5000)", min_value=0.0, max_value=5000.0, value=1000.0)
num_web_purchases = st.number_input("Number of Web Purchases (0-100)", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases (0-100)", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits (0-50)", min_value=0, max_value=50, value=10)
recency = st.number_input("Recency (days since last purchase) (0-365)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")