import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
CHARTS = os.path.join(BASE, "charts")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

# 1. Cleaned data load karo
train = pd.read_csv(os.path.join(DATA, "train_clean.csv"))
test = pd.read_csv(os.path.join(DATA, "test_clean.csv"))

# 2. TF-IDF: words ko numbers me badlo
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                        stop_words="english")
X_train = tfidf.fit_transform(train["clean"])
X_test = tfidf.transform(test["clean"])
y_train, y_test = train["label"], test["label"]

# 3. Teen models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
}

results = {}
best_name, best_f1, best_model, best_pred = None, 0, None, None

for name, model in models.items():
    print("Training", name, "...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    print(results[name])

    # Confusion matrix chart
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "cm_" + name.replace(" ", "_") + ".png"))
    plt.close()

    if results[name]["f1"] > best_f1:
        best_name, best_f1 = name, results[name]["f1"]
        best_model, best_pred = model, pred

# 4. Comparison chart
df = pd.DataFrame(results).T
df.plot(kind="bar", figsize=(8, 4), ylim=(0.8, 1))
plt.title("Model Comparison")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "model_comparison.png"))
plt.close()

# 5. Error analysis: best model ki galtiyan
wrong = test[best_pred != y_test].copy()
wrong["predicted"] = best_pred[best_pred != y_test]
wrong[["text", "label", "predicted"]].head(20).to_csv(
    os.path.join(CHARTS, "errors.csv"), index=False)

# 6. Best model save karo
joblib.dump(best_model, os.path.join(MODELS, "model.pkl"))
joblib.dump(tfidf, os.path.join(MODELS, "tfidf.pkl"))
with open(os.path.join(MODELS, "metrics.json"), "w") as f:
    json.dump({"best": best_name, "results": results}, f, indent=2)

print("\nBest model:", best_name)
print("Saved model, charts and metrics")