import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
from nltk.corpus import stopwords

st.title("📩 Spam Detection System")

# Load dataset
data = pd.read_csv("data/spam.csv")

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

data['text'] = data['text'].apply(preprocess)

# Vectorization
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
X = tfidf.fit_transform(data['text'])
y = data['label']

# Train model once
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# UI
user_input = st.text_area("Enter your message")

if st.button("Predict"):
    cleaned = preprocess(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == "spam":
        st.error("🚨 This message is SPAM")
    else:
        st.success("✅ This message is NOT SPAM")
