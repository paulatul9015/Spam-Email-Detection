# Spam Email Detection

A simple Python project for detecting whether an email is spam or ham using a text classification model.

## Features
- Preprocesses email text
- Builds a TF-IDF + Logistic Regression classifier
- Saves trained model and vectorizer
- Includes a small Streamlit-style interface and CLI prediction helper

## Project structure
- `app.py` — demo interface
- `src/train_model.py` — train and save the model
- `src/model_utils.py` — preprocessing and prediction utilities
- `data/sample_emails.csv` — example dataset

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
python app.py
```

## Example dataset format
The training CSV should contain at least these columns:
- `text`
- `label`

Where `label` is either `spam` or `ham`.

## License
This project is provided for educational/demo purposes.
