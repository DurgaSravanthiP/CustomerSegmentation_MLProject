import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

.main {
    background-color: #ffffff;
    padding: 0px;
    border-radius: 12px;
    box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.08);
}

h1 {
    color: #2c3e50;
    text-align: center;
}

.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    padding: 10px;
    font-size: 16px;
    border-radius: 8px;
    border: none;
}

.stButton>button:hover {
    background-color: #43a047;
}

.result-box {
    background-color: #e8f5e9;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    border-left: 6px solid #4CAF50;
}

.info-box {
    background-color: #e3f2fd;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    border-left: 6px solid #2196F3;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model and Scaler
# -----------------------------
kmeans = joblib.load("kmeans_customer_segmentation_model.pkl")
scaler = joblib.load("scaler_customer_segmentation.pkl")

# -----------------------------
# Cluster Names
# -----------------------------
cluster_names = {
    0: "High-Income, High-Spending Active Customers",
    1: "Older, Stable Customers with Balanced Spending",
    2: "Young Digital Shoppers (Online-Focused)",
    3: "Inactive and Low-Engagement Customers",
    4: "Low-Income, Low-Spending Occasional Buyers",
    5: "High-Income, Low-Spending Potential Premium Customers"
}

# -----------------------------
# Business Recommendations
# -----------------------------
cluster_recommendations = {
    0: "Focus on loyalty programs, premium memberships, and exclusive offers.",
    1: "Maintain engagement with personalized services and consistent communication.",
    2: "Target with digital marketing, app notifications, and online-only deals.",
    3: "Run re-engagement campaigns such as discounts and reminder emails.",
    4: "Provide budget-friendly products and seasonal discount campaigns.",
    5: "Upsell premium products and provide personalized recommendations."
}

# -----------------------------
# App UI
# -----------------------------
st.title("🧠 Customer Segmentation Predictor")
st.write("Fill in customer details to identify the customer segment.")

age = st.number_input("Age (18–100)", 18, 100, 30)
income = st.number_input("Income (0–200000)", 0.0, 200000.0, 50000.0)
total_spending = st.number_input("Total Spending (0–5000)", 0.0, 5000.0, 1000.0)
num_web_purchases = st.number_input("Web Purchases (0–100)", 0, 100, 10)
num_store_purchases = st.number_input("Store Purchases (0–100)", 0, 100, 10)
num_web_visits = st.number_input("Web Visits / Month (0–50)", 0, 50, 10)
recency = st.number_input("Recency (Days since last purchase)", 0, 365, 30)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Customer Segment"):
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
    cluster = kmeans.predict(input_scaled)[0]

    st.markdown(
        f"""
        <div class="result-box">
        <h3>Predicted Segment</h3>
        <p><strong>Cluster {cluster}</strong> – {cluster_names[cluster]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-box">
        <strong>Business Insight:</strong><br>
        {cluster_recommendations[cluster]}
        </div>
        """,
        unsafe_allow_html=True
    )
