#🧠 Customer Segmentation Using K-Means Clustering

This project applies unsupervised machine learning to segment customers based on their demographics, spending habits, and purchasing behavior.
It also includes a Streamlit web application that predicts the customer’s segment using a trained K-Means model.

#🚀 Features

Segments customers into 6 meaningful groups

Performs data cleaning, feature engineering, and EDA

Uses Elbow Method to determine the optimal number of clusters

Visualizes customer groups using PCA scatter plot

Saves trained model and scaler for reuse (.pkl files)

Streamlit app for real-time cluster prediction

#🛠 Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-Learn (KMeans, StandardScaler, PCA)

Joblib (Model Saving)

Streamlit (Web App)

#📂 Files Included

Analysis_Model.ipynb → Full data analysis, clustering, PCA visualization

segmentation.py → Streamlit app to predict customer cluster

customer_segmentation.csv → Dataset used

kmeans_customer_segmentation_model.pkl → Saved trained KMeans model

scaler_customer_segmentation.pkl → Saved StandardScaler

##📊 Project Workflow
#1️⃣ Data Preprocessing

Removed missing values

Converted dates

Cleaned and formatted columns

#2️⃣ Feature Engineering

Created new features:

Age

Total Spending

Total Children

Customer Since (days)

#3️⃣ Exploratory Data Analysis

Plots include:

Age distribution

Income distribution

Total spending

Boxplots

Correlation heatmap

#4️⃣ K-Means Clustering

Selected 7 key features

Standardized using StandardScaler

Optimal k = 6 using Elbow Method

Evaluated with Silhouette Score

#5️⃣ PCA Visualization

Reduced dataset to 2 components for visual cluster separation.

#6️⃣ Model Saving

Saved with joblib to reuse in the Streamlit app.

#🌐 Running the Streamlit App
▶ Install dependencies
pip install -r requirements.txt

▶ Run the app
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

App returns:
👉 Predicted Customer Segment (Cluster 0 – 5)

#📁 Project Structure
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


#🎯 Results

Best number of clusters: 6

Clear separation shown by PCA visualization

Meaningful customer groups for targeted marketing

Fully working prediction application

#👩‍💻 Author

Durga Sravanthi Peddoju
B.Tech CSE
SRM University AP
