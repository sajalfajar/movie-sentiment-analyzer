from datasets import load_dataset
import matplotlib.pyplot as plt
import os

# Dataset download (IMDB 50,000 reviews)
ds = load_dataset("stanfordnlp/imdb")
train = ds["train"].to_pandas()
test = ds["test"].to_pandas()

# CSV me save karo
os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

# Basic info
print("Train size:", train.shape)
print("Test size:", test.shape)
print("\nLabel count (0 = negative, 1 = positive):")
print(train["label"].value_counts())
print("\nSample review:")
print(train["text"].iloc[0][:300])

# Chart: positive vs negative
train["label"].value_counts().sort_index().plot(
    kind="bar", color=["red", "green"]
)
plt.xticks([0, 1], ["Negative", "Positive"], rotation=0)
plt.title("Class Distribution (Train)")
plt.ylabel("Reviews")
plt.savefig("charts/class_distribution.png")
print("\nChart saved in charts folder")
