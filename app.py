import os
from pathlib import Path

import streamlit as st

from src.model_utils import load_model_bundle, predict_email

MODEL_DIR = Path("model_artifacts")
MODEL_PATH = MODEL_DIR / "spam_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"


def main():
    st.title("Spam Email Detection")
    st.caption("Enter an email message to classify it as spam or ham.")

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        st.warning("Model not found. Train the model first by running `python src/train_model.py`.")
        return

    model, vectorizer = load_model_bundle(MODEL_PATH, VECTORIZER_PATH)

    email_text = st.text_area("Email content", height=250, placeholder="Type or paste the email here...")

    if st.button("Check email") and email_text.strip():
        prediction, probability = predict_email(model, vectorizer, email_text)
        confidence = max(probability)

        label_color = "green" if prediction == "ham" else "red"
        st.markdown(f"### Result: <span style='color:{label_color};'>{prediction.upper()}</span>", unsafe_allow_html=True)
        st.write(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
