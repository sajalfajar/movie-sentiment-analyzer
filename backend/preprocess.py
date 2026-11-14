import os
import re
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")


def clean_text(text):
    text = text.lower()                      # sab chhota (small) letters
    text = re.sub(r"<.*?>", " ", text)       # html tags hatao (<br />)
    text = re.sub(r"[^a-z\s]", " ", text)    # numbers aur symbols hatao
    text = re.sub(r"\s+", " ", text).strip() # extra spaces hatao
    return text


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))

    train["clean"] = train["text"].apply(clean_text)
    test["clean"] = test["text"].apply(clean_text)

    train.to_csv(os.path.join(DATA, "train_clean.csv"), index=False)
    test.to_csv(os.path.join(DATA, "test_clean.csv"), index=False)

    print("Before:", train["text"].iloc[0][:150])
    print("After :", train["clean"].iloc[0][:150])
    print("Cleaning done")