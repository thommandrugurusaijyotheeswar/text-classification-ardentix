# 📩 Text Classification System (Spam Detection)

This project is part of the **Ardentix AI/ML Engineer Intern selection assignment**.  
It demonstrates an end-to-end **Machine Learning pipeline** for text classification using Python.

The system classifies input text messages as **Spam** or **Ham (Not Spam)** and provides a simple web interface using **Streamlit**.

---

## 🚀 Project Overview

The project covers:
- Text preprocessing and feature extraction
- Training and comparing multiple ML models
- Model evaluation using standard metrics
- Saving/loading trained models
- Deploying a user-friendly web interface

---



## 📊 Dataset

- Dataset: **Spam Detection Dataset**
- Format: CSV file with two columns:
  - `label`: spam / ham
  - `text`: message content
- Stored at: `data/spam.csv`

---

## 🧹 Text Preprocessing

The following preprocessing steps are applied:
- Lowercasing text
- Removing punctuation and special characters
- Stopwords removal
- TF-IDF vectorization

---

## 🤖 Models Used

Two machine learning models were trained and compared:
- **Naive Bayes**
- **Logistic Regression**

### Final Model Used for Deployment
- **Naive Bayes**
- Chosen for simplicity and strong performance on text data

---

## 📈 Model Evaluation

The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

A comparison chart is generated and saved in the `outputs/` folder.

---

## 💾 Model Persistence

- Trained model is saved using **pickle**
- TF-IDF vectorizer is saved for reuse
- Stored in the `saved_models/` directory

This allows the Streamlit app to load the model without retraining.

---

## 🖥️ Web Interface (Streamlit)

A simple Streamlit-based UI allows users to:
- Enter a text message
- Get real-time predictions (Spam / Ham)

### Run the App Locally
```bash
python -m streamlit run app.py