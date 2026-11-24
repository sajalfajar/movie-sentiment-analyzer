import os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "models", "model.pkl"))
tfidf = joblib.load(os.path.join(BASE, "models", "tfidf.pkl"))

words = tfidf.get_feature_names_out()
coef = model.coef_[0]
order = coef.argsort()

print("TOP NEGATIVE WORDS:", list(words[order[:15]]))
print("TOP POSITIVE WORDS:", list(words[order[-15:][::-1]]))

err = pd.read_csv(os.path.join(BASE, "charts", "errors.csv"))
print("\nSAMPLE ERRORS (label 0=neg, 1=pos):")
for _, r in err.head(5).iterrows():
    print("-", r["text"][:200], "| actual:", r["label"], "| predicted:", r["predicted"])