---
title: "Code Tutorial: Simple Sentiment Analysis with Python"
date: 2026-08-16T10:00:00+05:30
draft: false
description: "A complete hands-on tutorial for building a sentiment analysis system in Python. Covers data loading, preprocessing, feature extraction, model training, evaluation, and comparison of multiple approaches."
tags: ["NLP", "Sentiment Analysis", "Python", "Tutorial", "Scikit-Learn"]
categories: ["NLP"]
image: "/images/blogs/pool-nlp/1.jpg"
keywords: ["sentiment analysis tutorial", "python NLP tutorial", "text classification python", "VADER sentiment", "scikit-learn text classification"]
---

Theory is essential, but code is where the learning truly happens. In this tutorial, we will build a complete sentiment analysis pipeline from scratch using Python. We will compare three approaches: rule-based (VADER), classical ML (TF-IDF + classifiers), and a brief look at using pre-trained transformers.

By the end, you will have a working sentiment analysis system and a clear understanding of the tradeoffs between different approaches.

---

### Step 1: Environment Setup

```python
# Install required packages
# pip install nltk scikit-learn pandas matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.pipeline import Pipeline
import nltk
import re
import warnings

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
```

---

### Step 2: Generating a Dataset

For this tutorial, we will create a synthetic movie review dataset. In practice, you would use real datasets like IMDB, Yelp, or Amazon reviews.

```python
# Synthetic dataset for demonstration
positive_reviews = [
    "This movie was absolutely fantastic and I loved every minute of it",
    "An incredible film with outstanding performances from the entire cast",
    "Beautiful cinematography and a deeply moving storyline",
    "One of the best movies I have ever seen in my life",
    "A masterpiece of modern cinema that will stand the test of time",
    "Brilliant acting and a script that keeps you on the edge of your seat",
    "I was blown away by how good this film turned out to be",
    "Heartwarming story with perfect pacing and wonderful characters",
    "An absolute delight from start to finish highly recommend",
    "Stunning visuals and an emotionally powerful narrative",
    "The director did an amazing job bringing this story to life",
    "Every scene was perfectly crafted and the ending was superb",
    "A truly enjoyable experience that exceeded all my expectations",
    "Wonderful performances and a gripping plot that never lets up",
    "This is cinema at its finest with great writing and acting",
    "I laughed I cried and I left the theater feeling inspired",
    "A beautiful and touching film that resonates deeply",
    "Exceptional storytelling with memorable and relatable characters",
    "The best film of the year without question",
    "A riveting and emotionally charged movie experience",
]

negative_reviews = [
    "This movie was terrible and a complete waste of my time",
    "Awful acting and a plot that made absolutely no sense at all",
    "I was bored out of my mind for the entire two hours",
    "One of the worst films I have ever had the misfortune to watch",
    "Terrible script with wooden dialogue and flat characters",
    "A disappointing mess that failed to deliver on its promise",
    "I wanted to walk out of the theater halfway through",
    "Poorly directed with cheap special effects and bad editing",
    "An absolute disaster of a film that should never have been made",
    "The acting was cringe worthy and the story was predictable",
    "I cannot believe how bad this movie turned out to be",
    "Painfully slow with no redeeming qualities whatsoever",
    "A waste of talented actors on a horrible script",
    "The worst movie experience I have had this entire year",
    "Confusing plot holes and characters that I could not care about",
    "Dull boring and completely forgettable from start to finish",
    "I regret spending money on this awful excuse for a movie",
    "Nothing about this film worked and I am deeply disappointed",
    "An embarrassingly bad attempt at serious filmmaking",
    "This movie is proof that big budgets cannot save bad writing",
]

# Combine into a dataset
texts = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

# Create DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
```

---


![Diagram showing NLP model architecture and data flow](/images/blogs/pool-nlp/8.jpg)

### Step 3: Text Preprocessing

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# Keep negation words — they are critical for sentiment
negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                  'nowhere', 'nor', 'cannot', "can't", "won't", "don't",
                  "doesn't", "didn't", "wasn't", "weren't", "hasn't",
                  "haven't", "hadn't", "isn't", "aren't"}
stop_words = stop_words - negation_words

def preprocess_text(text):
    """Clean and preprocess text for sentiment analysis."""
    # Lowercase
    text = text.lower()

    # Remove special characters but keep apostrophes
    text = re.sub(r"[^a-zA-Z'\s]", '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stop words (keeping negation words)
    words = text.split()
    words = [w for w in words if w not in stop_words]

    return ' '.join(words)

# Apply preprocessing
df['text_clean'] = df['text'].apply(preprocess_text)

print("Original:", df['text'].iloc[0])
print("Cleaned:", df['text_clean'].iloc[0])
```

---

### Step 4: Approach 1 — Rule-Based (VADER)

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    """Get VADER sentiment prediction."""
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 1  # Positive
    elif scores['compound'] <= -0.05:
        return 0  # Negative
    else:
        return -1  # Neutral (we will treat as negative for binary)

# Apply VADER to all texts
df['vader_pred'] = df['text'].apply(vader_sentiment)
# Treat neutral as negative for binary comparison
df['vader_pred_binary'] = df['vader_pred'].apply(lambda x: max(x, 0))

vader_accuracy = accuracy_score(df['label'], df['vader_pred_binary'])
print(f"\nVADER Accuracy: {vader_accuracy:.4f}")
print("\nVADER Classification Report:")
print(classification_report(
    df['label'], df['vader_pred_binary'],
    target_names=['Negative', 'Positive']
))
```

---

### Step 5: Approach 2 — TF-IDF + Classical ML


![Illustration of text analysis and language understanding techniques](/images/blogs/pool-nlp/7.jpg)

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'],
    test_size=0.3, random_state=42, stratify=df['label']
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Define multiple pipelines to compare
pipelines = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ('clf', LinearSVC(C=1.0, max_iter=1000, random_state=42))
    ]),
}

# Train and evaluate each pipeline
results = {}
for name, pipeline in pipelines.items():
    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, df['text_clean'], df['label'], cv=5, scoring='accuracy'
    )

    # Train on train set, evaluate on test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy
    }

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
```

---

### Step 6: Feature Analysis

Understanding which words drive the model's predictions:

```python
# Use Logistic Regression for interpretability
lr_pipeline = pipelines['Logistic Regression']
tfidf = lr_pipeline.named_steps['tfidf']
clf = lr_pipeline.named_steps['clf']

# Get feature names and their weights
feature_names = tfidf.get_feature_names_out()
coefficients = clf.coef_[0]

# Top positive and negative features
top_positive_idx = np.argsort(coefficients)[-15:]
top_negative_idx = np.argsort(coefficients)[:15]

print("Top 15 Positive Indicators:")
for idx in reversed(top_positive_idx):
    print(f"  {feature_names[idx]:>25s}: {coefficients[idx]:>8.4f}")

print("\nTop 15 Negative Indicators:")
for idx in top_negative_idx:
    print(f"  {feature_names[idx]:>25s}: {coefficients[idx]:>8.4f}")
```

---

### Step 7: Confusion Matrix Visualization

```python
from sklearn.metrics import ConfusionMatrixDisplay

best_pipeline = pipelines['Logistic Regression']
y_pred = best_pipeline.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Negative', 'Positive'],
    cmap='Blues',
    ax=ax
)
plt.title('Sentiment Analysis — Confusion Matrix')
plt.tight_layout()
plt.savefig("sentiment_confusion_matrix.png", dpi=150)
plt.show()
```

---


![Visual representation of natural language processing pipeline](/images/blogs/pool-nlp/6.jpg)

### Step 8: Comparing All Approaches

```python
# Summary comparison
comparison = pd.DataFrame({
    'Approach': ['VADER (Rule-Based)'] + list(results.keys()),
    'Accuracy': [vader_accuracy] + [r['test_accuracy'] for r in results.values()],
})

print("\n" + "="*50)
print("FINAL COMPARISON")
print("="*50)
print(comparison.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
bars = plt.barh(comparison['Approach'], comparison['Accuracy'], color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
plt.xlabel('Accuracy')
plt.title('Sentiment Analysis — Approach Comparison')
plt.xlim(0, 1.0)
for bar, acc in zip(bars, comparison['Accuracy']):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{acc:.3f}', va='center')
plt.tight_layout()
plt.savefig("approach_comparison.png", dpi=150)
plt.show()
```

---

### Step 9: Making Predictions on New Text

```python
def predict_sentiment(text, pipeline):
    """Predict sentiment for a new text."""
    cleaned = preprocess_text(text)
    prediction = pipeline.predict([cleaned])[0]

    # Get probability if available
    if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = max(proba)
    else:
        confidence = None

    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

# Test with new reviews
new_reviews = [
    "What an incredible film! I was completely captivated.",
    "I fell asleep twice during this boring movie.",
    "It had some good moments but overall was mediocre.",
    "The special effects were stunning but the plot was weak.",
    "A perfect movie night choice, absolutely loved it!",
]

print("\nPredictions on new reviews:")
print("-" * 60)
for review in new_reviews:
    sentiment, confidence = predict_sentiment(review, pipelines['Logistic Regression'])
    conf_str = f" ({confidence:.2f})" if confidence else ""
    print(f"{sentiment:>8}{conf_str}: {review}")
```

---

### Step 10: Key Takeaways

This tutorial demonstrated three important things:

1. **VADER is a strong baseline for social media text** — no training required, handles informal language well, but struggles with nuanced or domain-specific text.

2. **TF-IDF + Logistic Regression is remarkably effective** for sentiment analysis. It is fast, interpretable, and works well with moderate amounts of training data.

3. **Feature analysis reveals what the model learned** — inspecting the top features confirms that the model captured meaningful sentiment indicators.

For production systems with larger datasets, consider fine-tuning a pre-trained transformer model (DistilBERT, RoBERTa) for the best accuracy. But always start with these simpler approaches as baselines — they are often good enough and much cheaper to deploy.

---

### Final Thoughts

Sentiment analysis is one of the most accessible entry points into NLP. With just a few lines of Scikit-Learn code, you can build a system that understands human opinions and emotions in text.

In the next post, we begin a new chapter entirely: we dive into the deep learning revolution that transformed NLP forever. We start with **The Paper That Started It All: Attention Is All You Need**.
