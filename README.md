# Lab Assignment 4 – NLP Preprocessing and Text Classification

## Setup
```bash
pip install -r requirements.txt
```

## Run as Python script
```bash
python nlp_text_classification.py
```

## Run as Jupyter Notebook
```bash
jupyter notebook nlp_text_classification.ipynb
```

## Project Structure
```
nlp_project/
├── nlp_text_classification.py      # Main script
├── nlp_text_classification.ipynb   # Jupyter notebook
├── requirements.txt
├── README.md
└── results.png                     # Generated after running
```

## What's Covered
1. **Tokenization** – word_tokenize from NLTK
2. **Stopword Removal** – NLTK English stopwords
3. **Stemming** – Porter Stemmer
4. **Lemmatization** – WordNet Lemmatizer
5. **CountVectorizer** – Bag-of-words + bigrams
6. **TF-IDF Vectorizer** – Sublinear TF scaling + bigrams
7. **Naive Bayes** – Multinomial (Count + TF-IDF)
8. **Logistic Regression** – L2 regularized
9. **Linear SVM** – LinearSVC
10. **Metrics** – Accuracy, F1, Classification Report, Confusion Matrix
11. **Sklearn Pipeline** – End-to-end best practice
