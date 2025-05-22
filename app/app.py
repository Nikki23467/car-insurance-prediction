import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Car Insurance Predictor", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
    body {
        color: white;
        background-color: #0e1117;
    }
    .main {
        background-color: #0e1117;
    }
    .big-card {
        background: linear-gradient(135deg, #1c1f26 0%, #1e1f2f 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        color: white;
        background-color: #e50914;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff1e27;
        color: white;
    }
    .quote-box {
        background-color: #262730;
        padding: 1rem;
        border-left: 4px solid #e50914;
        border-radius: 8px;
        font-style: italic;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1c1f26;
        color: #bbb;
        text-align: center;
        padding: 0.75rem 0;
        font-size: 0.9rem;
        border-top: 1px solid #333;
        user-select: none;
        z-index: 1000;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def load_users():
    if os.path.exists("app/users.csv"):
        return pd.read_csv("app/users.csv")
    else:
        return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    df = load_users()
    if username in df.username.values:
        return False
    hashed_pw = make_hashes(password)
    new_row = pd.DataFrame([[username, hashed_pw]], columns=["username", "password"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("app/users.csv", index=False)
    return True

@st.cache_resource
def load_model():
    model = joblib.load("app/model.pkl")
    preprocessor = joblib.load("app/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

# --- Sidebar ---
st.sidebar.title("🔐 Account")
auth_choice = st.sidebar.radio("Choose an action", ["Login", "Register"])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    if auth_choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            users_df = load_users()
            user = users_df[users_df['username'] == username]
            if not user.empty and check_hashes(password, user.iloc[0]['password']):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Invalid credentials")
    else:
        new_user = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type='password')
        if st.sidebar.button("Register"):
            if save_user(new_user, new_password):
                st.sidebar.success("Account created successfully!")
            else:
                st.sidebar.error("Username already exists")

    # Inspirational Quote
    st.sidebar.markdown("""<div class='quote-box'>
        🚗 "Data is the new fuel — especially in car insurance prediction."
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown("Developed by **Nikhilesh K**")

# --- Main Content ---
st.markdown(f"<div class='big-card'>"
            f"<h1>🚘 Car Insurance Predictor</h1>"
            f"<h3>Your personal AI-powered insurance advisor</h3>"
            f"<p>This app uses machine learning to predict whether a customer is likely to purchase car insurance.</p>"
            f"<p>It helps insurance companies streamline customer outreach and target the right audience.</p>"
            f"</div>", unsafe_allow_html=True)

if st.session_state.logged_in:
    st.success(f"Welcome, {st.session_state['username']}!")

    menu = st.selectbox("📂 Navigation", [
        "View Train/Test Data",
        "Model Accuracy",
        "Upload for Prediction",
        "Prediction Ratio",
        "Download Datasets",
        "View Users"
    ])

    if menu == "View Train/Test Data":
        st.subheader("📊 Training Data")
        st.dataframe(train_df)
        st.subheader("🧪 Testing Data")
        st.dataframe(test_df)

    elif menu == "Model Accuracy":
        st.subheader("📈 Accuracy Scores")
        st.markdown("""
        - Logistic Regression: **87.50%**
        - XGBoost Classifier: **87.48%**
        """)

    elif menu == "Upload for Prediction":
        st.subheader("📤 Upload a CSV file for prediction")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            X = preprocessor.transform(input_df)
            preds = model.predict(X)
            input_df["Prediction"] = np.where(preds == 1, "Interested", "Not Interested")
            st.dataframe(input_df)

    elif menu == "Prediction Ratio":
        st.subheader("📊 Prediction Distribution")
        X = preprocessor.transform(test_df)
        preds = model.predict(X)
        counts = pd.Series(preds).map({1: "Interested", 0: "Not Interested"}).value_counts()
        st.write(counts)
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#00b4d8", "#f15bb5"])
        ax.axis("equal")
        st.pyplot(fig)

    elif menu == "Download Datasets":
        st.subheader("⬇️ Download Files")
        with open("data/train.csv", "rb") as f:
            st.download_button("Download Train Data", f, file_name="train.csv")
        with open("data/test.csv", "rb") as f:
            st.download_button("Download Test Data", f, file_name="test.csv")
        with open("data/sample_submission.csv", "rb") as f:
            st.download_button("Download Sample Submission", f, file_name="sample_submission.csv")

    elif menu == "View Users":
        st.subheader("👥 Registered Users")
        st.dataframe(load_users())
else:
    st.warning("Please log in to access the full dashboard.")

# --- Footer ---
st.markdown("""
<div class="footer">
    © 2025 Car Insurance Predictor | All rights reserved. <br>Created by <strong>Nikhilesh K</strong>
</div>
""", unsafe_allow_html=True)
