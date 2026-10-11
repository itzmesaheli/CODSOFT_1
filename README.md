# CODSOFT Task 1: Movie Genre Classification

A machine learning model that predicts a movie's genre from its title and plot summary, built for the **CodSoft Machine Learning Internship**.

**Author:** Saheli

## Results

Evaluated on the 54,200-movie held-out test set (27 genres):

| Metric | Score |
|---|---|
| Accuracy | **60.99%** |
| Macro-F1 | 0.365 |

Accuracy is strong for a 27-class problem where a few genres (drama, documentary, comedy) dominate. Macro-F1 is lower because it weights every genre equally, and the model is weaker on rare genres such as war, news and reality-tv.

## How it works

1. **Load** the data (`ID ::: TITLE ::: GENRE ::: DESCRIPTION` format).
2. **Clean** the text: lowercase, remove URLs, punctuation and numbers.
3. **Vectorize** with TF-IDF (unigrams + bigrams, English stop words removed, sublinear term frequency, up to 150,000 features).
4. **Compare three classifiers** on a stratified 80/20 validation split:
   - Complement Naive Bayes
   - Logistic Regression
   - Linear SVM
5. **Select** the model with the best validation macro-F1, then retrain it on the full training data.
6. **Predict** the test set, save the predictions, and score them against the provided solution file.

## Project structure

```
CODSOFT_1/
├── movie_genre_classifier.py   # full training + evaluation + prediction script
├── confusion_matrix.png        # normalized confusion matrix (validation split)
├── test_predictions.csv        # predicted genre for every test movie
├── requirements.txt
└── README.md
```

## Dataset

IMDb Genre Classification Dataset (linked in the CodSoft task sheet, also available on Kaggle). Download it and place these files in a `data/` folder:

```
data/
├── train_data.txt
├── test_data.txt
└── test_data_solution.txt
```

The dataset files are not included in this repo because of their size.

## How to run

```bash
pip install -r requirements.txt

# train, evaluate and generate predictions
python movie_genre_classifier.py

# predict the genre of any plot
python movie_genre_classifier.py --predict "A detective hunts a serial killer in a rainy city."
```

Outputs are written to an `output/` folder: the trained model (`genre_model.joblib`), `confusion_matrix.png` and `test_predictions.csv`.

It also runs in Google Colab with no installs: upload the script, create a `data/` folder with the three dataset files, and run `!python movie_genre_classifier.py`.

## Possible improvements

- Use `class_weight="balanced"` to improve results on rare genres.
- Try word embeddings or a fine-tuned transformer such as DistilBERT.
- Turn this into a multi-label problem, since many movies belong to more than one genre.

## Tech stack

Python, pandas, scikit-learn, matplotlib, joblib

## Demo

LinkedIn video: *add your link here after posting* #codsoft
