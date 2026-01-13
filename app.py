import streamlit as st
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="centered"
)

# ----------------------------------
# Load Model & Vectorizer
# ----------------------------------
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_models", "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


# ----------------------------------
# Sidebar Navigation
# ----------------------------------
page = st.sidebar.selectbox(
    "📌 Select Page",
    ["Spam Detection", "Model Evaluation"]
)

# ----------------------------------
# Text Preprocessing
# ----------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    return text

# ----------------------------------
# PAGE 1: SPAM DETECTION
# ----------------------------------
if page == "Spam Detection":

    st.title("📩 Spam Detection System")
    st.write("Enter a message to check whether it is **Spam** or **Ham**.")

    user_input = st.text_area("✉ Enter your message")

    if st.button("🔍 Predict"):
        if user_input.strip() == "":
            st.warning("⚠ Please enter a message")
        else:
            processed_text = preprocess(user_input)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]

            if prediction == "spam":
                st.error("🚨 This message is **SPAM**")
            else:
                st.success("✅ This message is **NOT SPAM (HAM)**")

# ----------------------------------
# PAGE 2: MODEL EVALUATION
# ----------------------------------
if page == "Model Evaluation":

    st.title("📊 Model Evaluation - Confusion Matrix")

    # Load dataset
    data = pd.read_csv("data/spam.csv")

    data["text"] = data["text"].apply(preprocess)

    X = vectorizer.transform(data["text"])
    y = data["label"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
