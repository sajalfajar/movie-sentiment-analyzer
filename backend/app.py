import os
import json
import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from preprocess import clean_text

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
CHARTS = os.path.join(BASE, "charts")
FRONTEND = os.path.join(os.path.dirname(BASE), "frontend")

# Saved model load karo
model = joblib.load(os.path.join(MODELS, "model.pkl"))
tfidf = joblib.load(os.path.join(MODELS, "tfidf.pkl"))
with open(os.path.join(MODELS, "metrics.json")) as f:
    metrics = json.load(f)

app = FastAPI(title="Movie Sentiment Analyzer")


class Review(BaseModel):
    text: str


@app.post("/predict")
def predict(review: Review):
    clean = clean_text(review.text)
    vec = tfidf.transform([clean])
    proba = model.predict_proba(vec)[0]
    label = int(proba.argmax())
    return {
        "sentiment": "Positive" if label == 1 else "Negative",
        "confidence": round(float(proba[label]) * 100, 1),
    }


@app.get("/api/metrics")
def get_metrics():
    return metrics


# Charts aur website serve karo (ye sabse neeche rahe)
app.mount("/charts", StaticFiles(directory=CHARTS), name="charts")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")