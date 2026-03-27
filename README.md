# 🎵 Spotify Hit Predictor & Data Analysis

This is an interactive web application built with Python and **Streamlit** that analyzes Spotify tracks and predicts whether a song will be a "HIT" based on its audio features.

## 🚀 Features
* **Data Cleaning & Preprocessing:** Handles missing values and removes outliers (e.g., in tempo).
* **Exploratory Data Analysis (EDA):** Interactive statistics, correlation matrices, and visualizations using Matplotlib.
* **Machine Learning Models:**
  * **K-Means Clustering:** Groups songs into 3 clusters based on audio features.
  * **Logistic Regression:** Predicts if a song is a hit (popularity > 70) with confusion matrix and classification reports (accuracy, precision, recall).
  * **Multiple Linear Regression (Statsmodels):** Evaluates how well audio features explain track popularity.

## 🛠️ Technologies Used
* Python
* Streamlit
* Pandas & NumPy
* Scikit-Learn
* Statsmodels
* Matplotlib

## 💻 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`