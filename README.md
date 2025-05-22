🚗 Car Insurance Purchase Prediction
This project is a web-based machine learning application that predicts whether a customer is likely to purchase car insurance based on various demographic and behavioral features. It is built using Python, Streamlit, and machine learning models like Logistic Regression and XGBoost (Boosted Decision Tree).

🔗 Try it out: https://car-insurance-prediction-nikhilesh.streamlit.app/

📌 Table of Contents
Overview

Features

Tech Stack

Installation

Usage

Project Structure

Model Details

Screenshots

License

📖 Overview
This project, titled "A Study on Car Insurance Purchase Prediction using Two-Class Logistic Regression and Two-Class Boosted Decision Tree", aims to:

Analyze customer data and predict the likelihood of car insurance purchase.

Build an interactive Streamlit web app for real-time predictions.

Provide insights into customer segmentation and model evaluation metrics.

✅ Features
Upload test data as .csv for batch prediction.

Visual comparison of predictions from Logistic Regression and XGBoost.

User-friendly UI built with Streamlit.

Trained models and pipeline saved using joblib for fast inference.

🧰 Tech Stack
Language: Python 3.13+

IDE: Jupyter Notebook, PyCharm

Web App: Streamlit

ML Libraries: scikit-learn, XGBoost, pandas, numpy, seaborn, matplotlib

Model Deployment: Streamlit Cloud

💻 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/car-insurance-prediction.git
cd car-insurance-prediction
Create virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run locally:

bash
Copy
Edit
streamlit run app.py
🚀 Usage
Upload a CSV file (test.csv) in the required format.

View predictions from both models.

Analyze graphs and model performance metrics.

Download the results (if implemented).

🗂️ Project Structure
bash
Copy
Edit
├── app.py                    # Main Streamlit application
├── model_training.ipynb      # Jupyter Notebook for training and evaluation
├── models/
│   ├── logistic_model.pkl
│   └── xgboost_model.pkl
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── requirements.txt
└── README.md
🧠 Model Details
1. Logistic Regression
Simple, interpretable, and works well with linear decision boundaries.

Evaluated using accuracy, precision, recall, and confusion matrix.

2. XGBoost (Boosted Decision Tree)
More complex, ensemble-based model.

Handles non-linearity and interactions better.

Typically yields higher accuracy.

Both models are trained on a dataset with features such as:

Age, Gender, Region_Code

Driving License, Previously Insured

Vehicle Age & Damage

Annual Premium, Policy Sales Channel

Vintage
