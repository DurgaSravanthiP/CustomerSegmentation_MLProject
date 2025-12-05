# 🧠 Customer Segmentation Using K-Means Clustering

This project applies **unsupervised machine learning** to segment customers based on their demographics, spending habits, and purchasing behavior.  
It also includes a **Streamlit web application** that predicts a customer’s segment using a trained K-Means model.

---

## 🚀 Features

- Segments customers into **6 meaningful groups**
- Performs **data cleaning**, **feature engineering**, and **EDA**
- Uses **Elbow Method** to determine the optimal number of clusters
- Visualizes clusters using **PCA scatter plot**
- Saves trained model and scaler as **.pkl files**
- Streamlit app for **real-time cluster prediction**

---

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn (KMeans, StandardScaler, PCA)  
- Joblib (Model Saving)  
- Streamlit (Web App)

---

## 📂 Files Included

- **Analysis_Model.ipynb** → Data analysis, feature engineering, clustering, PCA visualization  
- **segmentation.py** → Streamlit app to predict customer segment  
- **customer_segmentation.csv** → Dataset used  
- **kmeans_customer_segmentation_model.pkl** → Trained KMeans model  
- **scaler_customer_segmentation.pkl** → Saved StandardScaler  
- **MLPPT_CustomerSegmentation.pdf** → Project PPT  
- **MLProjectReport_CustomerSegmentation.pdf** → Full project report  

---

## 📊 Project Workflow

### 1️⃣ Data Preprocessing
- Removed missing values  
- Converted date columns  
- Cleaned and formatted dataset  

### 2️⃣ Feature Engineering
Created new features:
- Age  
- Total Spending  
- Total Children  
- Customer Since (days)  

### 3️⃣ Exploratory Data Analysis
Visualized:
- Age distribution  
- Income distribution  
- Total spending  
- Boxplots  
- Correlation heatmap  

### 4️⃣ K-Means Clustering
- Selected **7 important features**  
- Standardized data using **StandardScaler**  
- Determined **optimal k = 6** using the Elbow Method  
- Evaluated clusters using **Silhouette Score**

### 5️⃣ PCA Visualization
- Reduced data to **2 components**  
- Displayed cluster separation using a PCA scatter plot  

### 6️⃣ Model Saving
Saved using joblib:
- `kmeans_customer_segmentation_model.pkl`  
- `scaler_customer_segmentation.pkl`

---

## 🌐 Running the Streamlit App

### ▶ Install dependencies

pip install -r requirements.txt
▶ Run the application
streamlit run segmentation.py

▶ App Functionality

User enters:

Age

Income

Total Spending

Web Purchases

Store Purchases

Web Visits

Recency

The app displays:
👉 Predicted Customer Segment (Cluster 0 – 5)

📁 Project Structure
ML_CustomerSegmentation/
│
├── Analysis_Model.ipynb

├── segmentation.py

├── customer_segmentation.csv

├── kmeans_customer_segmentation_model.pkl

├── scaler_customer_segmentation.pkl

├── README.md

├── MLPPT_CustomerSegmentation.pdf

└── MLProjectReport_CustomerSegmentation.pdf


🎯 Results
---
Optimal number of clusters: 6

PCA scatter plot shows clear separation of clusters

Identified meaningful customer groups for targeted marketing

Fully functional Streamlit prediction application


👩‍💻 Author
---
Durga Sravanthi Peddoju

B.Tech CSE

SRM University AP

