import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import pandas as pd

nltk.download('stopwords')

st.title("📩 Spam Detection System")
st.write("AI/ML Internship Assignment – Ardentix")

# Load and train model (for demo)
data = pd.read_csv("data/spam.csv")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

data['text'] = data['text'].apply(preprocess)

tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
X = tfidf.fit_transform(data['text'])
y = data['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# User input
user_input = st.text_area("Enter message")

if st.button("Predict"):
    cleaned = preprocess(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == "spam":
        st.error("🚨 This message is SPAM")
    else:
        st.success("✅ This message is NOT SPAM")
