---
title: "Key Concept: Understanding Self-Attention in Transformers"
date: 2026-09-03T10:00:00+05:30
draft: false
description: "Self-attention is the mechanism that allows Transformers to understand context. This deep dive covers the complete mechanics of self-attention with worked examples, code, and intuition."
tags: ["Deep Learning", "Transformers", "Self-Attention", "NLP", "Python"]
categories: ["Deep Learning"]
image: "https://picsum.photos/seed/self-attention-explained-cover/1200/630"
keywords: ["self-attention", "transformer self-attention", "attention mechanism", "query key value", "positional encoding", "context modeling"]
---

We have discussed attention in broad terms. Now it is time to go deep. Self-attention is the single most important mechanism in modern AI, and understanding it thoroughly — not just conceptually, but mechanically — is essential for anyone serious about deep learning.

In this post, we will trace through self-attention step by step, with concrete numbers, code, and intuition at every stage.

---

### Part 1: What "Self" Means in Self-Attention

In the attention mechanism we discussed earlier, there are two distinct sequences: a source and a target. In cross-attention (used in translation), the decoder attends to the encoder's output — two different sequences.

**Self-attention** is when a sequence attends to **itself**. Each word in the sentence looks at every other word in the *same* sentence to build a context-aware representation.

This is powerful because meaning depends on context:
- "I went to the **bank** to deposit money." (Financial institution)
- "I sat on the **bank** of the river." (Riverbank)

Self-attention allows "bank" to gather information from "deposit" and "money" in the first sentence, and from "river" in the second, producing different representations for the same word.

---

### Part 2: The Complete Computation — Worked Example

Let us trace through self-attention for the sentence: "The cat sat."

#### Step 1: Input Embeddings

Each word is represented as a vector. For simplicity, we use 4 dimensions:

```python
import numpy as np

np.random.seed(42)

# Word embeddings (normally learned or from a pre-trained model)
embeddings = {
    'The': np.array([1.0, 0.0, 1.0, 0.0]),
    'cat': np.array([0.0, 1.0, 0.0, 1.0]),
    'sat': np.array([1.0, 1.0, 0.0, 0.0]),
}

# Stack into matrix X: shape (3, 4)
X = np.stack([embeddings['The'], embeddings['cat'], embeddings['sat']])
print("Input X:")
print(X)
```

#### Step 2: Compute Q, K, V

Three separate weight matrices project the input into Query, Key, and Value spaces:

```python
d_model = 4
d_k = 4  # Dimension of Q and K
d_v = 4  # Dimension of V

# Weight matrices (normally learned through training)
W_Q = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
], dtype=float) * 0.5

W_K = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=float) * 0.5

W_V = np.array([
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1]
], dtype=float) * 0.5

Q = X @ W_Q  # (3, 4)
K = X @ W_K  # (3, 4)
V = X @ W_V  # (3, 4)

print("\nQ (Queries):")
print(Q.round(3))
print("\nK (Keys):")
print(K.round(3))
print("\nV (Values):")
print(V.round(3))
```

Each row of Q represents what that word is "looking for." Each row of K represents what that word "offers." Each row of V represents the information that word carries.


![Illustration of gradient flow and model training dynamics](https://picsum.photos/seed/self-attention-explained-1/800/450)

#### Step 3: Compute Attention Scores

Multiply Q by K transposed to get a score for every word pair:

```python
scores = Q @ K.T  # (3, 3)
print("\nRaw attention scores:")
print(scores.round(3))
```

The score at position (i, j) measures how much word $i$ should attend to word $j$.

#### Step 4: Scale

Divide by $\sqrt{d_k}$ to prevent large values that would make softmax very peaked:

```python
scaled_scores = scores / np.sqrt(d_k)
print("\nScaled scores:")
print(scaled_scores.round(3))
```

#### Step 5: Apply Softmax

Convert scores to attention weights (probabilities that sum to 1 for each row):

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)
print("\nAttention weights:")
print(attention_weights.round(3))
print("\nRow sums (should be 1.0):", attention_weights.sum(axis=1).round(3))
```

Each row shows how much attention that word pays to every other word. Higher values mean more attention.

#### Step 6: Compute Output

Multiply attention weights by V to get the context-aware output:

```python
output = attention_weights @ V  # (3, 4)
print("\nSelf-attention output:")
print(output.round(3))
```

Each row of the output is a weighted combination of all Value vectors, where the weights reflect the relevance of each word to the current word. The output for "cat" now contains information from "The" and "sat" — it is context-aware.

---

### Part 3: Why Q, K, V — Not Just Similarity?

A natural question: why not just compute similarity between the raw embeddings? Why project into separate Q, K, V spaces?

The separate projections allow the model to learn **different notions of relevance** for different purposes:

- The **Query** asks: "What kind of context do I need?"
- The **Key** answers: "What kind of context can I provide?"
- The **Value** provides: "Here is my actual content."

By having separate learned projections, the model can learn that "cat" should query for nearby verbs (using Q), while "sat" should advertise itself as a verb (using K), and the actual information passed is the semantic content (using V).


![Diagram showing neural network layers and data transformation](https://picsum.photos/seed/self-attention-explained-2/800/450)

If we used the same vectors for both matching and content, these roles would be conflated.

---

### Part 4: Positional Encoding — The Missing Piece

Self-attention has no inherent sense of order. The sentence "cat the sat" would produce the same attention scores as "the cat sat" because the dot product is symmetric with respect to position.

**Positional encodings** inject position information into the embeddings. The original Transformer uses sinusoidal functions:

```python
def positional_encoding(seq_length, d_model):
    """Generate sinusoidal positional encodings."""
    PE = np.zeros((seq_length, d_model))

    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / 10000 ** (i / d_model))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / 10000 ** (i / d_model))

    return PE

# Generate positional encodings for our example
PE = positional_encoding(3, 4)
print("Positional Encodings:")
print(PE.round(3))

# Add to embeddings
X_with_pos = X + PE
print("\nEmbeddings + Positional Encoding:")
print(X_with_pos.round(3))
```

The sinusoidal encoding has a useful property: the model can learn to attend to relative positions. The encoding for position $p + k$ can be expressed as a linear function of the encoding at position $p$, which allows the model to learn relative offsets.

Modern models often use **learned positional embeddings** (a trainable embedding for each position) or **rotary positional embeddings (RoPE)**, which inject position information directly into the attention computation.

---

### Part 5: Causal (Masked) Self-Attention

For decoder models (GPT, Claude), self-attention must be **causal** — each position can only attend to itself and previous positions. This is enforced by adding a mask:

```python
def causal_self_attention(X, W_Q, W_K, W_V):
    """Self-attention with causal masking."""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # Apply causal mask
    seq_len = X.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    scores = scores + mask

    weights = softmax(scores)
    output = weights @ V

    return output, weights

output_causal, weights_causal = causal_self_attention(X, W_Q, W_K, W_V)
print("Causal attention weights:")
print(weights_causal.round(3))
```

Notice that the upper triangle of the weight matrix is zero — each position can only attend to previous positions and itself.

---

### Part 6: Multi-Head Self-Attention — Full Implementation


![Visual representation of deep learning network architecture](https://picsum.photos/seed/self-attention-explained-3/800/450)

```python
def multi_head_self_attention(X, n_heads, d_model, causal=False):
    """Complete multi-head self-attention."""
    assert d_model % n_heads == 0
    d_k = d_model // n_heads
    seq_len = X.shape[0]

    all_heads = []

    for h in range(n_heads):
        # Each head has its own projections
        W_Q = np.random.randn(d_model, d_k) * (d_k ** -0.5)
        W_K = np.random.randn(d_model, d_k) * (d_k ** -0.5)
        W_V = np.random.randn(d_model, d_k) * (d_k ** -0.5)

        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

        scores = Q @ K.T / np.sqrt(d_k)

        if causal:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores = scores + mask

        weights = softmax(scores)
        head_output = weights @ V
        all_heads.append(head_output)

    # Concatenate heads
    concatenated = np.concatenate(all_heads, axis=-1)  # (seq_len, d_model)

    # Output projection
    W_O = np.random.randn(d_model, d_model) * (d_model ** -0.5)
    output = concatenated @ W_O

    return output

# Example with 4 heads
d_model = 64
X_demo = np.random.randn(10, d_model)  # 10 tokens, 64 dimensions
output = multi_head_self_attention(X_demo, n_heads=4, d_model=d_model, causal=True)
print(f"Input shape: {X_demo.shape}")
print(f"Output shape: {output.shape}")
```

---

### Part 7: What Self-Attention Learns

Research has shown that different attention heads in trained Transformers learn different linguistic patterns:

- **Syntactic heads:** Attend to syntactically related words (subject-verb, adjective-noun).
- **Positional heads:** Attend to nearby positions (local context).
- **Coreference heads:** Attend to the antecedent of pronouns ("he" → "John").
- **Semantic heads:** Attend to semantically related words.
- **Delimiter heads:** Attend to special tokens (sentence boundaries, [CLS] token).

This specialization emerges naturally from training — no one explicitly programs these behaviors.

---

### Part 8: The Information Flow Perspective

Another way to think about self-attention: it is a **communication mechanism** between positions.

Before self-attention, each position only knows its own embedding. After self-attention, each position has gathered information from every other position. After multiple layers of self-attention, each position has a representation that incorporates information from the entire sequence, processed through multiple levels of abstraction.

Layer 1: Local relationships (adjacent words).
Layer 6: Global relationships (sentence-level meaning).
Layer 12: Abstract semantic understanding.

This progressive refinement through stacked self-attention layers is what gives Transformers their representational power.

---

### Final Thoughts

Self-attention is the foundation of everything in modern AI. The Query-Key-Value framework, softmax normalization, positional encoding, causal masking, and multi-head parallelism — these are the building blocks of every transformer model.

Understanding self-attention at the mathematical level, not just the conceptual level, is what separates someone who *uses* AI tools from someone who *builds* them.

In the next and final post of this section, we look at the remarkable generalization of Transformers beyond text: **Vision Transformers and Beyond**.
