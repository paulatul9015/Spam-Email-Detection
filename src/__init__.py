from pathlib import Path

import joblib
import numpy as np


def load_model_bundle(model_path: str | Path, vectorizer_path: str | Path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_email(model, vectorizer, email_text: str):
    transformed = vectorizer.transform([email_text])
    prediction = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]
    return prediction, probabilities


def score_text(model, vectorizer, email_text: str):
    prediction, probabilities = predict_email(model, vectorizer, email_text)
    probability = float(np.max(probabilities))
    return prediction, probability
