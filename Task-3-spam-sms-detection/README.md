# Spam SMS Detection

This project builds an AI model to classify SMS messages as **spam** or **legitimate (ham)** using three machine learning algorithms:
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

## Dataset
I use the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle.

## Features
- Text preprocessing using TF-IDF vectorization.
- Training and evaluation of 3 different models.
- Saving trained models and vectorizer for later use.

## Files
- `models/naive_bayes_model.pkl`
- `models/logistic_regression_model.pkl`
- `models/svm_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `requirements.txt`
- `spam_sms_detection.ipynb`

## Usage
```python
import joblib

model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

sample_messages = ["Win a free iPhone!", "Hey, what time is our meeting?"]
X = vectorizer.transform(sample_messages)
preds = model.predict(X)
print(preds)  # [1 0]
