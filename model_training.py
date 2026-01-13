import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("data/spam.csv")

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['text'] = data['text'].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Vectorization
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}


results = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, pos_label="spam"),
        "Recall": recall_score(y_test, preds, pos_label="spam"),
        "F1 Score": f1_score(y_test, preds, pos_label="spam")
    }

# Display results
results_df = pd.DataFrame(results).T
print(results_df)

# Visualization
results_df.plot(kind="bar", figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.show()




# Create directory if needed
os.makedirs("saved_models", exist_ok=True)

# Save Naive Bayes model
with open("saved_models/model.pkl", "wb") as f:
    pickle.dump(models["Naive Bayes"], f)

# Save TF-IDF vectorizer
with open("saved_models/vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Naive Bayes model and vectorizer saved successfully")

