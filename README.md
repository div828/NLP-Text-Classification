# NLP Text Classification Project

## Objective
To implement NLP preprocessing and build a text classification model.

## Features
- Tokenization
- Stopword removal
- Stemming & Lemmatization
- TF-IDF and CountVectorizer
- Machine Learning Models (NB, SVM, Logistic Regression)

## Dataset
20 Newsgroups Dataset

## Results
Best Accuracy: ~88%

## Conclusion
Naive Bayes model performed best for this classification task.
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
