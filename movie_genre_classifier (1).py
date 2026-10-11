"""
CODSOFT Task 1 - Movie Genre Classification
-------------------------------------------
Predicts a movie's genre from its plot summary using TF-IDF features and
Naive Bayes / Logistic Regression / Linear SVM.

Expected files in ./data/  (IMDb "Genre Classification Dataset" format):
    train_data.txt          ID ::: TITLE ::: GENRE ::: DESCRIPTION
    test_data.txt           ID ::: TITLE ::: DESCRIPTION
    test_data_solution.txt  ID ::: TITLE ::: GENRE ::: DESCRIPTION   (optional)

Usage:
    pip install pandas scikit-learn matplotlib joblib
    python movie_genre_classifier.py                      # train + evaluate + predict
    python movie_genre_classifier.py --predict "A detective hunts a serial killer in a rainy city."
"""

import argparse
import re
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = Path("data")
OUT_DIR = Path("output")
MODEL_PATH = OUT_DIR / "genre_model.joblib"
SEP = " ::: "


# ---------------------------------------------------------------- data loading
def load_file(path: Path, has_genre: bool) -> pd.DataFrame:
    """Parse a ' ::: ' separated file into a DataFrame."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(SEP)
            if has_genre and len(parts) >= 4:
                rows.append((parts[0], parts[1], parts[2].strip(), SEP.join(parts[3:])))
            elif not has_genre and len(parts) >= 3:
                rows.append((parts[0], parts[1], None, SEP.join(parts[2:])))
    return pd.DataFrame(rows, columns=["id", "title", "genre", "description"])


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_pipeline(clf) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=150_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("clf", clf),
        ]
    )


# ------------------------------------------------------------------- training
def train_and_evaluate() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Step 1: load
    train = load_file(DATA_DIR / "train_data.txt", has_genre=True)
    print(f"[1] Loaded {len(train):,} training rows, {train['genre'].nunique()} genres")
    print(train["genre"].value_counts().head(10), "\n")

    train["clean"] = (train["title"] + " " + train["description"]).map(clean_text)

    # Step 2: validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        train["clean"],
        train["genre"],
        test_size=0.2,
        random_state=42,
        stratify=train["genre"],
    )

    # Step 3: compare models
    candidates = {
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10),
        "Linear SVM": LinearSVC(C=0.5),
    }
    results = {}
    print("[2] Comparing models on validation split")
    for name, clf in candidates.items():
        start = time.time()
        pipe = make_pipeline(clf).fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro")
        results[name] = (f1, acc, pipe, pred)
        print(f"    {name:<24} accuracy={acc:.4f}  macro-F1={f1:.4f}  ({time.time() - start:.0f}s)")

    best_name = max(results, key=lambda k: results[k][0])
    _, _, _, best_val_pred = results[best_name]
    print(f"\n[3] Best model (by macro-F1): {best_name}\n")
    print(classification_report(y_val, best_val_pred, zero_division=0))

    # Step 4: confusion matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay.from_predictions(
        y_val, best_val_pred, ax=ax, xticks_rotation=90, normalize="true",
        values_format=".1f", colorbar=False,
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Step 5: refit best model on ALL training data and save
    final = make_pipeline(candidates[best_name]).fit(train["clean"], train["genre"])
    joblib.dump(final, MODEL_PATH)
    print(f"[4] Final model trained on all data and saved to {MODEL_PATH}")

    # Step 6: predict the test set
    test_path = DATA_DIR / "test_data.txt"
    if test_path.exists():
        test = load_file(test_path, has_genre=False)
        test["clean"] = (test["title"] + " " + test["description"]).map(clean_text)
        test["predicted_genre"] = final.predict(test["clean"])
        test[["id", "title", "predicted_genre"]].to_csv(
            OUT_DIR / "test_predictions.csv", index=False
        )
        print(f"[5] Saved {len(test):,} predictions to {OUT_DIR / 'test_predictions.csv'}")

        sol_path = DATA_DIR / "test_data_solution.txt"
        if sol_path.exists():
            sol = load_file(sol_path, has_genre=True)
            merged = test[["id", "predicted_genre"]].merge(sol[["id", "genre"]], on="id")
            acc = accuracy_score(merged["genre"], merged["predicted_genre"])
            f1 = f1_score(merged["genre"], merged["predicted_genre"], average="macro")
            print(f"[6] TEST SET  accuracy={acc:.4f}  macro-F1={f1:.4f}")


# ----------------------------------------------------------------- prediction
def predict_one(text: str) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Run the script without --predict first.")
    model = joblib.load(MODEL_PATH)
    print("Predicted genre:", model.predict([clean_text(text)])[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODSOFT Task 1 - Movie Genre Classification")
    parser.add_argument("--predict", metavar="PLOT", help="Predict the genre of a plot summary")
    args = parser.parse_args()

    if args.predict:
        predict_one(args.predict)
    else:
        train_and_evaluate()
