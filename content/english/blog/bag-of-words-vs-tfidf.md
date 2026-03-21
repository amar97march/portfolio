---
title: "Classic NLP: Bag-of-Words vs. TF-IDF Explained"
date: 2026-08-10T10:00:00+05:30
draft: false
description: "Bag-of-Words and TF-IDF are the foundational text representation methods in NLP. Learn how they work, their differences, and when they are still the right choice in modern applications."
tags: ["NLP", "TF-IDF", "Bag of Words", "Text Processing", "Python"]
categories: ["NLP"]
image: "https://images.unsplash.com/photo-1543286386-2e659306cd6c?w=1200&h=630&fit=crop&auto=format"
keywords: ["bag of words", "TF-IDF", "text representation", "document term matrix", "text vectorization", "NLP feature extraction"]
---

Before there were word embeddings and transformers, NLP practitioners needed a way to convert text into numbers. The two foundational methods — **Bag-of-Words (BoW)** and **TF-IDF** — are deceptively simple, surprisingly effective, and still widely used today.

Understanding these methods is essential because they reveal the core challenge of text representation: how do you capture the meaning of a document in a fixed-length numerical vector?

---

### Part 1: Bag-of-Words — Counting Words

The Bag-of-Words model represents a document as a **vector of word counts**. The "bag" metaphor means we throw all the words into a bag, shake it up, and only care about *which* words are present and *how many times* — not their order.

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "The cat and the dog played"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

import pandas as pd
df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=[f"Doc {i+1}" for i in range(len(documents))]
)
print(df)
```

Output:
```
       and  cat  dog  log  mat  on  played  sat  the
Doc 1    0    1    0    0    1   1       0    1    2
Doc 2    0    0    1    1    0   1       0    1    2
Doc 3    1    1    1    0    0   0       1    0    2
```

Each row is a document, each column is a word, and each cell is the count of that word in that document.

---

### Part 2: The Problem with Raw Counts

Raw word counts have a major issue: **common words dominate.** The word "the" appears in every document with high frequency, but it carries almost no information about what the document is *about*.

Document 1 and Document 2 look very similar because they share "the," "sat," and "on" — but their actual subjects (cat vs. dog) are different.

This is where TF-IDF comes in.

![Document-term matrix showing word counts across multiple text documents](https://picsum.photos/seed/bag-of-words-vs-tfidf-1/800/450)

---

### Part 3: TF-IDF — Weighing Words by Importance

**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**. It weighs each word by two factors:

#### Term Frequency (TF)
How often does this word appear in *this* document?

$$TF(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total words in } d}$$

Words that appear frequently in a document are likely important to that document.

#### Inverse Document Frequency (IDF)
How rare is this word across *all* documents?

$$IDF(t) = \log\frac{N}{1 + df(t)}$$

Where $N$ is the total number of documents and $df(t)$ is the number of documents containing word $t$.

Words that appear in many documents are less informative. "The" has a low IDF. "Astrophysics" has a high IDF.

#### TF-IDF Score

$$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$$

A word gets a high TF-IDF score if it appears frequently in a specific document but rarely across the corpus. This is a strong signal that the word is important to that document's topic.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)

df_tfidf = pd.DataFrame(
    X_tfidf.toarray().round(3),
    columns=tfidf.get_feature_names_out(),
    index=[f"Doc {i+1}" for i in range(len(documents))]
)
print(df_tfidf)
```

Now "the" has a lower weight (it appears everywhere), while distinguishing words like "mat," "log," and "played" have higher weights.

---

### Part 4: N-grams — Capturing Word Pairs

Both BoW and TF-IDF lose word order. The sentence "dog bites man" and "man bites dog" produce the same vector. **N-grams** partially address this by considering sequences of words:

- **Unigrams (n=1):** Individual words — "dog", "bites", "man"
- **Bigrams (n=2):** Word pairs — "dog bites", "bites man"
- **Trigrams (n=3):** Word triples — "dog bites man"

```python
# Using bigrams with TF-IDF
tfidf_bigram = TfidfVectorizer(ngram_range=(1, 2))
X_bigram = tfidf_bigram.fit_transform(documents)

print(f"Unigram features: {len(tfidf.get_feature_names_out())}")
print(f"Uni+Bigram features: {len(tfidf_bigram.get_feature_names_out())}")
print(f"\nBigram features include:")
bigram_features = [f for f in tfidf_bigram.get_feature_names_out() if ' ' in f]
print(bigram_features)
```

N-grams capture some context but at the cost of a much larger feature space.

---

### Part 5: Practical Considerations

#### Vocabulary Size Control
Large corpora can produce millions of unique words. Control the vocabulary:

```python
tfidf = TfidfVectorizer(
    max_features=10000,      # Keep only top 10K features
    min_df=5,                # Ignore words appearing in fewer than 5 documents
    max_df=0.95,             # Ignore words appearing in more than 95% of documents
    ngram_range=(1, 2),      # Unigrams and bigrams
    stop_words='english',    # Remove common English stop words
    sublinear_tf=True        # Apply log normalization to TF
)
```

#### Sparse Matrices
BoW and TF-IDF produce very sparse matrices (most entries are zero). Scikit-Learn stores them as sparse matrices to save memory:

```python
print(f"Shape: {X_tfidf.shape}")
print(f"Non-zero entries: {X_tfidf.nnz}")
print(f"Sparsity: {1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.2%}")
```

![TF-IDF weighting scheme highlighting important terms in documents](https://picsum.photos/seed/bag-of-words-vs-tfidf-2/800/450)

---

### Part 6: BoW/TF-IDF for Classification

Despite their simplicity, BoW and TF-IDF features work remarkably well with classical ML classifiers:

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Load a subset of the 20 Newsgroups dataset
categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

# Build pipelines with different classifiers
pipelines = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LinearSVC(max_iter=1000))
    ]),
}

for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, newsgroups.data, newsgroups.target, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

You will find that simple TF-IDF + Linear SVM achieves surprisingly strong performance on text classification tasks — often within a few percentage points of transformer-based models, while being orders of magnitude faster.

---

### Part 7: Document Similarity with TF-IDF

TF-IDF vectors enable document similarity computation using **cosine similarity**:

```python
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning is a branch of artificial intelligence",
    "Deep learning is a subset of machine learning",
    "The weather today is sunny and warm",
    "Neural networks are used in deep learning"
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(documents)

# Compute pairwise similarity
sim_matrix = cosine_similarity(X)

df_sim = pd.DataFrame(
    sim_matrix.round(3),
    columns=[f"Doc {i+1}" for i in range(len(documents))],
    index=[f"Doc {i+1}" for i in range(len(documents))]
)
print(df_sim)
```

Documents about ML/DL will have high similarity; the weather document will have low similarity to the others. This is the foundation of search engines and recommendation systems.

![Cosine similarity matrix comparing document vectors for text retrieval](https://picsum.photos/seed/bag-of-words-vs-tfidf-3/800/450)

---

### Part 8: Limitations

**What BoW/TF-IDF miss:**

1. **Word order:** "dog bites man" = "man bites dog" in BoW.
2. **Semantics:** "happy" and "joyful" are treated as completely different features.
3. **Context:** The same word always gets the same weight, regardless of context.
4. **Compositionality:** The meaning of a phrase is more than the sum of its words.

These are precisely the problems that word embeddings and transformers solve.

---

### Part 9: When to Still Use BoW/TF-IDF

Despite their limitations, BoW/TF-IDF remain valuable:

- **Baseline models:** Always start with TF-IDF + Logistic Regression. It is fast, and you need a baseline to beat.
- **Small datasets:** With fewer than 10,000 documents, TF-IDF + classical ML often outperforms fine-tuned transformers.
- **High-speed requirements:** TF-IDF vectorization and linear classifiers are extremely fast.
- **Interpretability:** You can inspect which words drive the classification.
- **Search and retrieval:** TF-IDF-based search (like Elasticsearch's BM25) remains the backbone of many search systems.

---

### Final Thoughts

Bag-of-Words and TF-IDF are the bedrock of text representation in NLP. They transform the messy, variable-length world of text into clean, fixed-length numerical vectors that machine learning algorithms can work with.

While modern approaches have moved to embeddings and transformers, understanding BoW and TF-IDF is essential. They are still used in production, they provide strong baselines, and they illuminate the fundamental challenge of representing meaning as numbers.

In the next post, we apply these concepts to one of NLP's most popular tasks: **Sentiment Analysis — Understanding Opinion Mining**.
