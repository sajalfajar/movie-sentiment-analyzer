import os
import json
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")

model = joblib.load(os.path.join(MODELS, "model.pkl"))
tfidf = joblib.load(os.path.join(MODELS, "tfidf.pkl"))

data = {
    "vocab": {w: int(i) for w, i in tfidf.vocabulary_.items()},
    "idf": [round(float(x), 5) for x in tfidf.idf_],
    "coef": [round(float(x), 5) for x in model.coef_[0]],
    "intercept": float(model.intercept_[0]),
    "stop_words": sorted(tfidf.get_stop_words()),
}

out = os.path.join(BASE, "..", "frontend", "model.json")
with open(out, "w") as f:
    json.dump(data, f, separators=(",", ":"))

print("model.json saved. Words:", len(data["vocab"]))