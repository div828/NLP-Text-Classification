"""
Lab Assignment 4: NLP Preprocessing and Text Classification
============================================================
Title: NLP Preprocessing and Text Classification
Objective: Implement NLP preprocessing techniques and build a text classification model
"""

# ============================================================
# SECTION 1: IMPORTS & SETUP
# ============================================================
import re
import string
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP Libraries
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.pipeline import Pipeline

print("=" * 60)
print("NLP PREPROCESSING AND TEXT CLASSIFICATION")
print("=" * 60)


# ============================================================
# SECTION 2: DATASET LOADING
# ============================================================
print("\n[1] LOADING DATASET")
print("-" * 40)

# Using 20 Newsgroups dataset (4 categories for simplicity)
categories = [
    'rec.sport.hockey',
    'sci.med',
    'comp.graphics',
    'talk.politics.misc'
]

train_data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)
test_data = fetch_20newsgroups(
    subset='test',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

print(f"Training samples : {len(train_data.data)}")
print(f"Testing  samples : {len(test_data.data)}")
print(f"Categories       : {train_data.target_names}")
print(f"\nSample text (first 300 chars):\n{train_data.data[0][:300]}")


# ============================================================
# SECTION 3: NLP PREPROCESSING
# ============================================================
print("\n[2] NLP PREPROCESSING")
print("-" * 40)

stop_words   = set(stopwords.words('english'))
stemmer      = PorterStemmer()
lemmatizer   = WordNetLemmatizer()


def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove special characters / numbers
    3. Tokenize
    4. Remove stopwords
    5. Stemming OR Lemmatization
    Returns: cleaned string
    """
    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove URLs, emails, numbers, punctuation
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 3 – tokenize
    tokens = word_tokenize(text)

    # Step 4 – remove stopwords & short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Step 5 – stemming or lemmatization
    if use_stemming:
        tokens = [stemmer.stem(t) for t in tokens]
    elif use_lemmatization:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


# Demo on a single sentence
sample = "The doctors are studying 3 new medical treatments for COVID-19 patients!"
print(f"Original  : {sample}")
print(f"Tokenized : {word_tokenize(sample.lower())}")
tokens_no_stop = [w for w in word_tokenize(sample.lower())
                  if w not in stop_words and w.isalpha()]
print(f"No stops  : {tokens_no_stop}")
print(f"Stemmed   : {[stemmer.stem(w) for w in tokens_no_stop]}")
print(f"Lemmatized: {[lemmatizer.lemmatize(w) for w in tokens_no_stop]}")
print(f"Cleaned   : {preprocess_text(sample)}")

# Preprocess all data
print("\nPreprocessing training and testing data …")
X_train_raw = train_data.data
X_test_raw  = test_data.data
y_train      = train_data.target
y_test       = test_data.target

X_train_clean = [preprocess_text(t) for t in X_train_raw]
X_test_clean  = [preprocess_text(t) for t in X_test_raw]
print("Done.")


# ============================================================
# SECTION 4: TEXT VECTORIZATION
# ============================================================
print("\n[3] TEXT VECTORIZATION")
print("-" * 40)

# --- CountVectorizer ---
count_vec = CountVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_count = count_vec.fit_transform(X_train_clean)
X_test_count  = count_vec.transform(X_test_clean)
print(f"CountVectorizer matrix : {X_train_count.shape}")
print(f"Sample features        : {count_vec.get_feature_names_out()[:10].tolist()}")

# --- TF-IDF Vectorizer ---
tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf_vec.fit_transform(X_train_clean)
X_test_tfidf  = tfidf_vec.transform(X_test_clean)
print(f"\nTF-IDF matrix          : {X_train_tfidf.shape}")

# Top TF-IDF terms per class
print("\nTop 10 TF-IDF terms per category:")
for i, cat in enumerate(train_data.target_names):
    idx   = np.where(y_train == i)[0]
    mean  = np.asarray(X_train_tfidf[idx].mean(axis=0)).flatten()
    top10 = np.argsort(mean)[-10:][::-1]
    terms = tfidf_vec.get_feature_names_out()[top10]
    print(f"  {cat:30s}: {', '.join(terms)}")


# ============================================================
# SECTION 5: MODEL TRAINING & EVALUATION
# ============================================================
print("\n[4] MODEL TRAINING AND EVALUATION")
print("-" * 40)

models = {
    'Naive Bayes (Count)':     (MultinomialNB(),         X_train_count,  X_test_count),
    'Naive Bayes (TF-IDF)':    (MultinomialNB(),         X_train_tfidf,  X_test_tfidf),
    'Logistic Regression':     (LogisticRegression(max_iter=1000, C=1.0),
                                 X_train_tfidf, X_test_tfidf),
    'Linear SVM':              (LinearSVC(max_iter=2000, C=1.0),
                                 X_train_tfidf, X_test_tfidf),
}

results = {}
for name, (clf, Xtr, Xte) in models.items():
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'accuracy': acc, 'f1': f1, 'clf': clf, 'y_pred': y_pred}
    print(f"  {name:30s} | Accuracy: {acc:.4f} | F1: {f1:.4f}")


# ============================================================
# SECTION 6: DETAILED EVALUATION OF BEST MODEL
# ============================================================
print("\n[5] DETAILED EVALUATION – BEST MODEL")
print("-" * 40)

best_name = max(results, key=lambda k: results[k]['accuracy'])
best      = results[best_name]
print(f"Best model: {best_name}\n")
print(classification_report(y_test, best['y_pred'],
                             target_names=train_data.target_names))


# ============================================================
# SECTION 7: VISUALIZATIONS
# ============================================================
print("\n[6] GENERATING VISUALIZATIONS …")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("NLP Text Classification – Results", fontsize=16, fontweight='bold')

# 7a  Model comparison (accuracy)
ax = axes[0, 0]
names = list(results.keys())
accs  = [results[n]['accuracy'] for n in names]
bars  = ax.barh(names, accs, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
ax.set_xlim(0, 1.05)
ax.set_xlabel("Accuracy")
ax.set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, accs):
    ax.text(acc + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{acc:.3f}', va='center', fontsize=10)

# 7b  Confusion matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=[c.split('.')[1] for c in train_data.target_names],
            yticklabels=[c.split('.')[1] for c in train_data.target_names])
ax.set_title(f"Confusion Matrix – {best_name}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# 7c  Class distribution
ax = axes[1, 0]
counts = pd.Series(y_train).value_counts().sort_index()
ax.bar([train_data.target_names[i].split('.')[-1] for i in counts.index],
       counts.values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
ax.set_title("Training Set Class Distribution")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=15)

# 7d  F1 scores per category (best model)
ax = axes[1, 1]
report = classification_report(y_test, best['y_pred'],
                                target_names=train_data.target_names,
                                output_dict=True)
cat_f1  = [report[c]['f1-score'] for c in train_data.target_names]
cat_lbl = [c.split('.')[1] for c in train_data.target_names]
ax.bar(cat_lbl, cat_f1, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
ax.set_ylim(0, 1.05)
ax.set_title(f"Per-Category F1 – {best_name}")
ax.set_xlabel("Category")
ax.set_ylabel("F1 Score")
ax.tick_params(axis='x', rotation=15)
for i, v in enumerate(cat_f1):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('/home/claude/nlp_project/results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results.png")


# ============================================================
# SECTION 8: PREDICT ON CUSTOM TEXT
# ============================================================
print("\n[7] PREDICT ON CUSTOM SENTENCES")
print("-" * 40)

best_clf = best['clf']

custom_texts = [
    "The hockey player scored a hat trick in the final game of the playoffs.",
    "The MRI scan revealed a tumor in the patient's brain requiring immediate surgery.",
    "OpenGL provides APIs for rendering 2D and 3D vector graphics on GPU hardware.",
    "The senator's speech on immigration policy sparked heated debate in Congress.",
]

for txt in custom_texts:
    cleaned  = preprocess_text(txt)
    vec      = tfidf_vec.transform([cleaned])
    pred_idx = best_clf.predict(vec)[0]
    pred_cat = train_data.target_names[pred_idx]
    print(f"  Text     : {txt[:70]}…")
    print(f"  Predicted: {pred_cat}\n")


# ============================================================
# SECTION 9: SKLEARN PIPELINE (BEST PRACTICE)
# ============================================================
print("\n[8] SKLEARN PIPELINE DEMO")
print("-" * 40)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                               sublinear_tf=True)),
    ('clf',   LinearSVC(max_iter=2000, C=1.0)),
])
pipeline.fit(X_train_clean, y_train)
pipe_preds = pipeline.predict(X_test_clean)
pipe_acc   = accuracy_score(y_test, pipe_preds)
print(f"Pipeline accuracy: {pipe_acc:.4f}")
print("Pipeline stages  :", [step[0] for step in pipeline.steps])


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, res in results.items():
    print(f"  {name:32s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")
print(f"\nBest model: {best_name}")
print(f"Accuracy  : {best['accuracy']:.4f}")
print("=" * 60)
