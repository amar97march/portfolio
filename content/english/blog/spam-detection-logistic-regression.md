---
title: "Use Case: Building a Spam Detector with Logistic Regression"
date: 2026-06-05T10:00:00+05:30
draft: false
description: "Build a complete spam detection system using logistic regression and TF-IDF — from text preprocessing to model evaluation, with production-ready code."
tags: ["Logistic Regression", "NLP", "Spam Detection", "Text Classification", "Scikit-Learn"]
categories: ["Machine Learning"]
image: "https://picsum.photos/seed/spam-detection-logistic-regression-cover/1200/630"
keywords: ["spam detector", "logistic regression NLP", "text classification", "TF-IDF", "email spam filter"]
---

Earlier in this series, I discussed spam filters as my favorite real-world example of machine learning. Now it is time to build one ourselves. We are going to construct a complete spam detection system using **logistic regression** and **TF-IDF** (Term Frequency-Inverse Document Frequency) — a classic and still remarkably effective approach to text classification.

This project brings together several concepts from previous posts: logistic regression for classification, feature engineering for text data, and proper model evaluation.

## The Problem

Given a text message or email, classify it as **spam** or **ham** (not spam). This is a binary classification problem — exactly what logistic regression was designed for.

## Setting Up

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import re
import string
```

## Creating a Realistic Dataset

```python
# Simulating a spam/ham dataset
np.random.seed(42)

spam_messages = [
    "Congratulations! You've won a $1000 gift card! Click here to claim NOW!",
    "URGENT: Your account has been compromised. Verify your identity immediately.",
    "FREE FREE FREE! Limited time offer on weight loss pills!",
    "You have been selected for a cash prize of $5,000,000!",
    "Act now! Exclusive deal just for you. 90% off luxury watches.",
    "Your loan has been approved! No credit check required. Apply today!",
    "Winner! You've been chosen for our monthly prize draw! Click here!",
    "Make $5000 weekly working from home! No experience needed!",
    "IMPORTANT: We noticed suspicious activity. Update your password now!",
    "Discount pharmacy - buy medications at 80% off! No prescription needed.",
    "You are our lucky winner! Claim your iPhone 15 Pro now!",
    "Double your investment in 30 days! Guaranteed returns! Act fast!",
    "Hot singles in your area want to meet you tonight!",
    "Your package is waiting! Pay $1.99 shipping to claim your reward!",
    "Bank security alert: unusual login detected. Click to secure account.",
]

ham_messages = [
    "Hey, can we reschedule our meeting to 3pm tomorrow?",
    "Thanks for sending the report. I'll review it tonight.",
    "The deployment pipeline is failing again. Can you check the logs?",
    "Happy birthday! Hope you have a wonderful day!",
    "Just confirming dinner reservation for Saturday at 7pm.",
    "Can you pick up some milk on your way home?",
    "The client approved the design mockups. We can start development.",
    "Running 10 minutes late to the standup. Start without me.",
    "Great presentation today. The stakeholders were impressed.",
    "Your order has been shipped. Estimated delivery: Tuesday.",
    "Let me know when you're free to discuss the project timeline.",
    "I attached the updated spreadsheet with Q3 numbers.",
    "Saw this article and thought of you. Check it out when you have time.",
    "The kids have a school event next Friday. Can you make it?",
    "Coffee tomorrow morning? I want to catch up on the new project.",
]

# Expand the dataset with variations
def generate_variations(messages, n_variations=20):
    expanded = []
    for _ in range(n_variations):
        for msg in messages:
            variation = msg
            if np.random.random() > 0.5:
                variation = variation.lower()
            if np.random.random() > 0.7:
                variation = variation.upper()
            words = variation.split()
            if np.random.random() > 0.6 and len(words) > 3:
                idx = np.random.randint(0, len(words))
                words.insert(idx, np.random.choice(['!!!', 'please', 'now', 'today']))
            expanded.append(' '.join(words))
    return expanded

spam_expanded = generate_variations(spam_messages, 40)
ham_expanded = generate_variations(ham_messages, 40)

texts = spam_expanded + ham_expanded
labels = [1] * len(spam_expanded) + [0] * len(ham_expanded)

df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset size: {len(df)}")
print(f"Spam: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"Ham: {len(df) - df['label'].sum()} ({(1-df['label'].mean())*100:.1f}%)")
```

## Step 1: Text Preprocessing

```python
def preprocess_text(text):
    """Clean and normalize text for classification"""
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', 'URL', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', 'EMAIL', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'PHONE', text)

    # Remove currency amounts
    text = re.sub(r'\$[\d,]+\.?\d*', 'MONEY', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply preprocessing
df['text_clean'] = df['text'].apply(preprocess_text)

# Feature engineering: extract text-level features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['uppercase_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
)
df['exclamation_count'] = df['text'].str.count('!')
df['has_url_pattern'] = df['text_clean'].str.contains('URL').astype(int)
df['has_money_pattern'] = df['text_clean'].str.contains('MONEY').astype(int)

# Compare spam vs ham characteristics
print("\nSpam characteristics:")
print(df[df['label']==1][['text_length', 'uppercase_ratio', 'exclamation_count']].mean())
print("\nHam characteristics:")
print(df[df['label']==0][['text_length', 'uppercase_ratio', 'exclamation_count']].mean())
```

![Preprocessing text data for spam classification](https://picsum.photos/seed/spam-detection-logistic-regression-1/800/450)

## Step 2: TF-IDF Vectorization

TF-IDF converts text into numerical features by weighing words based on their frequency in a document and their rarity across all documents.

```python
# Split the data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,       # Limit vocabulary size
    ngram_range=(1, 2),      # Use unigrams and bigrams
    min_df=2,                # Ignore very rare terms
    max_df=0.95,             # Ignore very common terms
    strip_accents='unicode',
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"Training matrix shape: {X_train_tfidf.shape}")

# Most important spam-indicative words
feature_names = tfidf.get_feature_names_out()
print(f"\nSample features: {feature_names[:20]}")
```

## Step 3: Train the Model

```python
# Logistic Regression with regularization
model = LogisticRegression(
    C=1.0,              # Regularization strength (lower = more regularization)
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs'
)

model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
```

## Step 4: Evaluate the Model

```python
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives (Ham correctly identified):  {cm[0][0]}")
print(f"  False Positives (Ham incorrectly flagged):   {cm[0][1]}")
print(f"  False Negatives (Spam that got through):     {cm[1][0]}")
print(f"  True Positives (Spam correctly caught):      {cm[1][1]}")

# AUC-ROC
auc = roc_auc_score(y_test, y_prob)
print(f"\nAUC-ROC: {auc:.4f}")

# For spam detection, false negatives (spam getting through) are annoying,
# but false positives (legitimate email marked as spam) are dangerous.
# We should optimize for HIGH PRECISION.
```

![Evaluating model performance with classification metrics](https://picsum.photos/seed/spam-detection-logistic-regression-2/800/450)

## Step 5: Analyze What the Model Learned

```python
# Most indicative words for spam vs ham
feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# Top spam indicators
spam_indicators = pd.DataFrame({
    'word': feature_names,
    'coefficient': coefficients
}).sort_values('coefficient', ascending=False)

print("\nTop 15 SPAM indicators:")
print(spam_indicators.head(15)[['word', 'coefficient']].to_string(index=False))

print("\nTop 15 HAM indicators:")
print(spam_indicators.tail(15)[['word', 'coefficient']].to_string(index=False))
```

## Step 6: Build a Pipeline

A scikit-learn Pipeline chains preprocessing and modeling into a single object:

```python
# Complete pipeline
spam_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )),
    ('classifier', LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced'
    ))
])

# Cross-validation with the pipeline
cv_scores = cross_val_score(spam_pipeline, df['text_clean'], df['label'],
                             cv=5, scoring='f1')
print(f"\n5-Fold CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Fit the final pipeline
spam_pipeline.fit(X_train_text, y_train)
```

## Step 7: Test with New Messages

```python
def classify_message(message, pipeline):
    """Classify a single message as spam or ham"""
    cleaned = preprocess_text(message)
    probability = pipeline.predict_proba([cleaned])[0]
    prediction = pipeline.predict([cleaned])[0]

    label = "SPAM" if prediction == 1 else "HAM"
    confidence = max(probability) * 100

    return label, confidence

# Test with new messages
test_messages = [
    "Hey, are we still on for lunch tomorrow?",
    "CONGRATULATIONS! You won $10,000! Click here to claim your prize NOW!!!",
    "The quarterly report is ready for your review.",
    "FREE iPhone giveaway! Be the lucky winner today!",
    "Can you send me the API documentation?",
    "Your account will be suspended unless you verify immediately!",
    "Great meeting today. Let's follow up next week.",
    "Make money fast! Work from home! No experience needed!!!",
]

print("\nSpam Detection Results:")
print("=" * 70)
for msg in test_messages:
    label, confidence = classify_message(msg, spam_pipeline)
    status = "CORRECT" if ("FREE" in msg or "CONGRATULATIONS" in msg or
                           "money fast" in msg or "suspended" in msg) == (label == "SPAM") else ""
    print(f"[{label}] ({confidence:.1f}%) {msg[:60]}...")
```

## Step 8: Threshold Tuning

```python
# For spam detection, we care more about precision (not marking ham as spam)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold for 99% precision (very few false positives)
target_precision = 0.99
for p, r, t in zip(precisions, recalls, thresholds):
    if p >= target_precision:
        print(f"Threshold: {t:.3f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall: {r:.4f}")
        break

# Apply high-precision threshold
high_precision_preds = (y_prob >= t).astype(int)
print(f"\nWith high-precision threshold:")
print(classification_report(y_test, high_precision_preds, target_names=['Ham', 'Spam']))
```

![Building a complete spam detection pipeline](https://picsum.photos/seed/spam-detection-logistic-regression-3/800/450)

## Production Considerations

If you were deploying this model, you would also need to consider:

1. **Updating the model regularly** as spam patterns evolve
2. **User feedback loops** — when users mark spam as not-spam (or vice versa), use that as training data
3. **Feature beyond text** — sender reputation, email headers, link analysis, attachment types
4. **Multiple model ensemble** — combine TF-IDF + logistic regression with other approaches for robustness
5. **Latency requirements** — the model must classify emails in milliseconds
6. **Handling adversarial attacks** — spammers intentionally try to evade filters

## Key Takeaways

1. **Logistic regression + TF-IDF** is a surprisingly strong baseline for text classification
2. **Preprocessing matters** — cleaning text, handling URLs, currency, and other patterns improves performance
3. **TF-IDF bigrams** capture important phrases like "click here" and "act now" that unigrams miss
4. **Coefficient analysis** reveals which words the model considers most spam-indicative
5. **Threshold tuning** lets you balance precision and recall for your specific use case
6. **Pipelines** keep preprocessing and modeling together, preventing data leakage

This project demonstrates that you do not need deep learning to build an effective text classifier. A well-engineered logistic regression model with thoughtful feature extraction can achieve excellent results.

---

*Next: A code tutorial walking through scikit-learn's linear regression API step by step.*
