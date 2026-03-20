---
title: "What are Embeddings? Turning Words into Numbers"
date: 2026-08-07T10:00:00+05:30
draft: false
description: "Word embeddings are the foundation of modern NLP. Learn how Word2Vec and GloVe turn words into dense vectors that capture meaning, and why this was a breakthrough for language AI."
tags: ["NLP", "Embeddings", "Word2Vec", "GloVe", "Deep Learning", "Python"]
categories: ["NLP"]
image: "/images/blogs/pool-nlp/1.jpg"
keywords: ["word embeddings", "Word2Vec", "GloVe", "word vectors", "semantic similarity", "NLP representations"]
---

Machine learning models operate on numbers. They cannot process the word "king" or "queen" directly. They need a numerical representation. But how do you convert a word into a number without losing its meaning?

For decades, the answer was simple but crude: one-hot encoding. Each word becomes a vector with a 1 in one position and 0s everywhere else. A vocabulary of 50,000 words means each word is a vector of length 50,000 with a single non-zero entry.

This representation has a fatal flaw: it treats every word as equally different from every other word. "Cat" is as different from "dog" as it is from "democracy." There is no notion of similarity or meaning.

**Word embeddings** changed everything.

---

### Part 1: The Core Idea

A word embedding is a dense, low-dimensional vector (typically 100-300 dimensions) where words with similar meanings are mapped to nearby points in the vector space.

In this space:
- "King" and "queen" are close together (both are royalty).
- "Cat" and "dog" are close together (both are pets/animals).
- "Cat" and "democracy" are far apart (unrelated concepts).

But it gets better. The *directions* in the space capture relationships:

$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$

The difference between "king" and "man" captures the concept of "royalty." Adding that to "woman" gives you "queen." The embedding space has learned abstract semantic relationships from raw text.

---

### Part 2: One-Hot Encoding vs. Dense Embeddings

| Property | One-Hot | Embeddings |
|----------|---------|-----------|
| **Dimensionality** | Vocabulary size (50,000+) | Small (100-300) |
| **Density** | Sparse (mostly zeros) | Dense (all values meaningful) |
| **Similarity** | All words equidistant | Similar words are close |
| **Relationships** | None captured | Semantic relationships encoded |
| **Memory** | Huge | Compact |

---

![Word vectors plotted in semantic space showing relationships](/images/blogs/pool-nlp/3.jpg)


### Part 3: Word2Vec — Learning Words from Context

Word2Vec, introduced by Tomas Mikolov at Google in 2013, was the breakthrough that popularized word embeddings. The key insight: **you can learn word meanings by looking at which words appear together.**

The linguistic hypothesis behind this is the **distributional hypothesis**: "You shall know a word by the company it keeps" (J.R. Firth, 1957).

Word2Vec has two architectures:

#### Skip-gram
Given a target word, predict the surrounding context words.

Input: "cat" → Predict: "the", "sat", "on", "mat"

The model learns a vector for "cat" that is useful for predicting its typical neighbors.

#### Continuous Bag of Words (CBOW)
Given the context words, predict the target word.

Input: "the", "sat", "on", "mat" → Predict: "cat"

Skip-gram tends to work better for rare words and small datasets. CBOW is faster and works well for frequent words.

```python
from gensim.models import Word2Vec

# Sample sentences (in practice, use millions of sentences)
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "rug"],
    ["cats", "and", "dogs", "are", "pets"],
    ["the", "king", "wore", "a", "crown"],
    ["the", "queen", "wore", "a", "tiara"],
    ["kings", "and", "queens", "rule", "kingdoms"],
]

# Train Word2Vec
model = Word2Vec(
    sentences,
    vector_size=50,    # Embedding dimensions
    window=3,          # Context window size
    min_count=1,       # Minimum word frequency
    sg=1,              # 1 = Skip-gram, 0 = CBOW
    epochs=100
)

# Get a word vector
cat_vector = model.wv['cat']
print(f"Vector for 'cat': {cat_vector[:10]}...")  # First 10 dimensions

# Find similar words
similar = model.wv.most_similar('cat', topn=3)
print(f"\nMost similar to 'cat': {similar}")
```

---

### Part 4: How Word2Vec Learns

Under the hood, Word2Vec is a shallow neural network with one hidden layer:

1. **Input:** A one-hot vector representing the target word.
2. **Hidden layer:** A weight matrix of size (vocabulary x embedding_size). This IS the embedding matrix.
3. **Output:** Predicted probabilities for context words.

During training, the model adjusts the weights to maximize the probability of the correct context words. After training, the hidden layer weights become the word embeddings.

The key insight is that the embeddings are a **byproduct** of training a prediction task. Word2Vec does not directly optimize for semantic similarity — it emerges naturally from learning to predict context.

#### Negative Sampling
Computing the output probability over the entire vocabulary (softmax over 50,000+ words) is computationally expensive. Word2Vec uses **negative sampling**: instead of updating all vocabulary weights, it only updates the target word and a small number of randomly sampled "negative" words. This makes training dramatically faster.

---

### Part 5: GloVe — Global Vectors

GloVe (Global Vectors for Word Representation), developed at Stanford in 2014, takes a different approach. Instead of predicting context words one at a time, GloVe builds a global co-occurrence matrix and factorizes it.

**The co-occurrence matrix** counts how often each pair of words appears together within a context window across the entire corpus.

GloVe optimizes the objective:

$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where $X_{ij}$ is the co-occurrence count and $f$ is a weighting function that prevents very frequent pairs from dominating.

GloVe combines the best of two worlds:
- **Global statistics** (like LSA): Uses the entire corpus's co-occurrence patterns.
- **Local context** (like Word2Vec): Captures word-level relationships.

---

![Word2Vec skip-gram architecture learning from context](/images/blogs/pool-nlp/4.jpg)


### Part 6: Using Pre-trained Embeddings

Training embeddings from scratch requires billions of words. Fortunately, pre-trained embeddings are freely available:

```python
import gensim.downloader as api

# Download pre-trained GloVe embeddings (trained on Wikipedia + Gigaword)
# This is ~66MB and contains 400,000 word vectors
glove = api.load("glove-wiki-gigaword-100")

# Explore semantic relationships
print("Most similar to 'king':")
print(glove.most_similar('king', topn=5))

print("\nking - man + woman =")
result = glove.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print(result)

# Compute similarity between two words
similarity = glove.similarity('cat', 'dog')
print(f"\nSimilarity between 'cat' and 'dog': {similarity:.4f}")

similarity2 = glove.similarity('cat', 'democracy')
print(f"Similarity between 'cat' and 'democracy': {similarity2:.4f}")
```

---

### Part 7: Visualizing Embeddings

High-dimensional embeddings can be visualized using dimensionality reduction:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Select words to visualize
words = ['king', 'queen', 'prince', 'princess',
         'man', 'woman', 'boy', 'girl',
         'cat', 'dog', 'fish', 'bird',
         'happy', 'sad', 'angry', 'calm']

# Get vectors
vectors = np.array([glove[w] for w in words])

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vectors_2d = tsne.fit_transform(vectors)

# Plot
plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], s=100)
    plt.annotate(word, (vectors_2d[i, 0] + 0.5, vectors_2d[i, 1] + 0.5), fontsize=12)

plt.title('Word Embeddings Visualized with t-SNE')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("word_embeddings_tsne.png", dpi=150)
plt.show()
```

You will see that semantically related words cluster together — royalty words near each other, animals near each other, emotions near each other.

---

![t-SNE visualization of word embeddings clustering by meaning](/images/blogs/pool-nlp/5.jpg)


### Part 8: Limitations of Static Embeddings

Word2Vec and GloVe produce **static embeddings**: each word has exactly one vector, regardless of context. This is a significant limitation:

- "Bank" has the same vector in "river bank" and "bank account."
- "Apple" has the same vector in "apple pie" and "Apple Inc."
- "Play" has the same vector in "play a game" and "watch a play."

This problem is solved by **contextual embeddings** — models like BERT and GPT that produce different representations for the same word depending on the surrounding context. We will cover these when we discuss transformers.

---

### Part 9: Embeddings Beyond Words

The embedding concept extends far beyond words:

- **Sentence embeddings:** Represent entire sentences as vectors (Sentence-BERT).
- **Document embeddings:** Represent entire documents (Doc2Vec).
- **Image embeddings:** Represent images as vectors (ResNet features, CLIP).
- **Graph embeddings:** Represent nodes in a graph (Node2Vec).
- **User embeddings:** Represent users for recommendation systems.

The core idea is always the same: map complex objects to dense vectors where similarity in vector space reflects real similarity.

---

### Final Thoughts

Word embeddings were a watershed moment in NLP. The idea that meaning could be represented as a direction in vector space — that arithmetic on vectors could capture semantic relationships — transformed the field.

While static embeddings have been largely superseded by contextual representations from transformers, understanding them is essential. They are the conceptual foundation on which all modern NLP is built.

In the next post, we will look at classical text representation methods: **Bag-of-Words vs. TF-IDF**.
