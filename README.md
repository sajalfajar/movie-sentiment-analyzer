# Sentiflix: Movie Review Sentiment Analyzer

An NLP project that classifies movie reviews as **positive** or **negative**.

**Made by:** Sajal Fajar
**Course:** NLP Course Project, Abasyn University

## Features
- Trained on 50,000 IMDB reviews
- Compares 3 models: Naive Bayes, Logistic Regression and SVM
- Best model: Logistic Regression (88.3% accuracy)
- Live website with confidence score
- Model Results page with charts and confusion matrices

## Tech Stack
Python, scikit-learn, TF-IDF, FastAPI, HTML, CSS, JavaScript

## Results
| Model | Accuracy | F1 |
|---|---|---|
| Naive Bayes | 0.8522 | 0.8481 |
| Logistic Regression | 0.8834 | 0.8837 |
| SVM | 0.8753 | 0.8737 |

## Project Structure
```
backend/    eda.py, preprocess.py, train.py, app.py, models, charts
frontend/   index.html, metrics.html, style.css, script.js
```

## How to Run Locally
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python eda.py
python preprocess.py
python train.py
cd ..
backend\venv\Scripts\python.exe -m uvicorn app:app --app-dir backend
```
Open http://127.0.0.1:8000

## Limitations
The model uses TF-IDF, so it ignores word order. Reviews like "terrible, but won me over" can be classified wrongly. Future work: BERT.