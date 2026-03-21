---
title: "Encoder vs. Decoder: The Two Halves of a Transformer"
date: 2026-08-28T10:00:00+05:30
draft: false
description: "The Transformer architecture has two halves: the encoder and the decoder. Learn how each works, why they differ, and how modern AI models choose one, the other, or both."
tags: ["Deep Learning", "Transformers", "BERT", "GPT", "Architecture"]
categories: ["Deep Learning"]
image: "https://images.unsplash.com/photo-1512486130939-2c4f79935e4f?w=1200&h=630&fit=crop&auto=format"
keywords: ["transformer encoder", "transformer decoder", "BERT vs GPT", "encoder decoder model", "masked self-attention", "cross-attention"]
---

The original Transformer has two major components: an **encoder** that reads and understands the input, and a **decoder** that generates the output. They share the same fundamental building blocks — attention, feed-forward networks, and residual connections — but they use them differently.

What is fascinating is that the most successful models since the original Transformer have chosen to use only *one half*. BERT uses only the encoder. GPT uses only the decoder. T5 uses both. Understanding why requires understanding what each half does and what it is good at.

---

### Part 1: The Encoder — Understanding the Input

The encoder's job is to produce a rich, context-aware representation of the input. It reads the entire input at once and creates an output where each position's representation incorporates information from every other position.

#### Architecture of One Encoder Layer

Each encoder layer has two sub-layers:

**1. Multi-Head Self-Attention**
Every position attends to every other position in the input. This is **bidirectional** — word 5 can look at both word 1 and word 10. This is crucial because understanding a word often requires context from both sides:

"The **bank** of the river was steep." (Context on the right disambiguates "bank.")

**2. Position-Wise Feed-Forward Network**
A two-layer fully connected network applied to each position independently:

$$FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

This adds non-linear transformation capacity. While attention handles interactions between positions, the FFN processes information within each position.

Both sub-layers use **residual connections** and **layer normalization**:

$$output = LayerNorm(x + SubLayer(x))$$

The original Transformer stacks 6 encoder layers. Each layer refines the representation, building increasingly abstract and context-rich features.

```
Input Embeddings + Positional Encoding
    ↓
[Self-Attention → Add & Norm → FFN → Add & Norm] × 6
    ↓
Encoder Output (one vector per input position)
```

---

### Part 2: The Decoder — Generating the Output

The decoder generates the output sequence one token at a time. It has a more complex structure because it must:
1. Attend to its own previously generated tokens.
2. Attend to the encoder's output.
3. Not "cheat" by looking at future tokens.

#### Architecture of One Decoder Layer

Each decoder layer has three sub-layers:

**1. Masked Multi-Head Self-Attention**
Similar to the encoder's self-attention, but with a crucial difference: **masking**. When generating token $t$, the decoder can only attend to tokens $1$ through $t-1$. It cannot look at future tokens (which have not been generated yet).

This masking is implemented by setting the attention scores for future positions to $-\infty$ before the softmax, which drives their attention weights to zero.

```python
import numpy as np

def create_causal_mask(seq_length):
    """Create a mask that prevents attending to future positions."""
    mask = np.triu(np.ones((seq_length, seq_length)), k=1)
    mask[mask == 1] = -np.inf
    return mask

mask = create_causal_mask(5)
print("Causal mask (0 = can attend, -inf = blocked):")
print(mask)
```

Output:
```
[[ 0. -inf -inf -inf -inf]
 [ 0.   0. -inf -inf -inf]
 [ 0.   0.   0. -inf -inf]
 [ 0.   0.   0.   0. -inf]
 [ 0.   0.   0.   0.   0.]]
```

Position 0 can only see itself. Position 2 can see positions 0, 1, and 2. Position 4 can see everything.

**2. Cross-Attention (Encoder-Decoder Attention)**
This is where the decoder "reads" the encoder's output. The Queries come from the decoder, while the Keys and Values come from the encoder.

This allows the decoder to focus on different parts of the input when generating each output token. For translation, when generating the French word for "cat," the cross-attention focuses on the English word "cat" in the encoder output.

**3. Position-Wise Feed-Forward Network**
Same structure as in the encoder.

```
Output Embeddings + Positional Encoding (shifted right)
    ↓
[Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm] × 6
    ↓
Linear → Softmax → Output Probabilities
```

---

![Transformer architecture with encoder and decoder blocks](https://picsum.photos/seed/encoder-vs-decoder-transformer-1/800/450)

### Part 3: Encoder-Only Models (BERT)

BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder. Why?

BERT's goal is **understanding**, not generation. It needs to produce a rich representation of the input text that can be used for downstream tasks (classification, NER, question answering).

**Pre-training objective — Masked Language Modeling (MLM):**
Randomly mask 15% of the input tokens and train the model to predict them. Because the encoder is bidirectional, it can use context from both sides of the masked token.

"The [MASK] sat on the mat." → Predict: "cat"

Both "The" (left context) and "sat on the mat" (right context) help predict the masked word.

**Why not use a decoder?** A decoder can only look left (due to the causal mask). This means it can only use partial context when making predictions. For understanding tasks, bidirectional context is strictly better.

**BERT-family models:** BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa.

**Best for:** Text classification, named entity recognition, question answering, semantic similarity, sentence embeddings.

---

### Part 4: Decoder-Only Models (GPT)

GPT (Generative Pre-trained Transformer) uses only the decoder. Why?

GPT's goal is **generation** — predicting the next token in a sequence. The causal mask is essential here because during generation, you genuinely do not know what comes next.

**Pre-training objective — Next Token Prediction:**
Given all previous tokens, predict the next one.

"The cat" → Predict: "sat"
"The cat sat" → Predict: "on"

This is an **autoregressive** process — each prediction becomes input for the next step.

**Why not use an encoder?** For generation, you do not have the complete input upfront. You build the output token by token. A decoder with causal masking naturally handles this.

**Why decoder-only for understanding tasks too?** It turns out that with enough scale and pre-training data, decoder-only models become surprisingly good at understanding tasks too. GPT-3 demonstrated that a large enough decoder can perform classification, translation, summarization, and question answering through prompting — without any architectural changes.

**GPT-family models:** GPT, GPT-2, GPT-3, GPT-4, Claude, Llama, Mistral.

**Best for:** Text generation, chatbots, code generation, general-purpose AI assistants.

---

### Part 5: Encoder-Decoder Models (T5, BART)

Some models use the full encoder-decoder architecture. T5 (Text-to-Text Transfer Transformer) treats every NLP task as a text-to-text problem:

- **Classification:** "classify: This movie is great" → "positive"
- **Translation:** "translate English to French: Hello" → "Bonjour"
- **Summarization:** "summarize: [long article]" → "[short summary]"

The encoder processes the full input bidirectionally, and the decoder generates the output autoregressively with cross-attention to the encoder.

**Encoder-decoder models:** T5, BART, mBART, Flan-T5.

**Best for:** Translation, summarization, question answering, any task with a clear input-output structure.

---

![Comparing encoder-only, decoder-only, and full transformer models](https://picsum.photos/seed/encoder-vs-decoder-transformer-2/800/450)

### Part 6: Comparison

| Aspect | Encoder-Only (BERT) | Decoder-Only (GPT) | Encoder-Decoder (T5) |
|--------|-------------------|-------------------|---------------------|
| **Attention** | Bidirectional | Causal (left-to-right) | Bidirectional (encoder) + Causal (decoder) |
| **Strength** | Understanding | Generation | Input-output mapping |
| **Pre-training** | Masked LM | Next token prediction | Span corruption / denoising |
| **Parallelism** | Full (encoding) | Partial (generation is sequential) | Full (encoding) + Partial (decoding) |
| **Scaling trend** | Plateaued ~1B params | Scales to 100B+ params | Moderate scaling |
| **Current dominance** | Specialized tasks | General-purpose AI | Translation, summarization |

---

### Part 7: The Scaling Laws Favor Decoders

An important trend in modern AI: **decoder-only models scale better.** As you add more parameters and data, decoder-only models continue to improve, while encoder-only and encoder-decoder models show diminishing returns.

This is likely because:
1. Next-token prediction is a rich training signal that captures both understanding and generation.
2. The autoregressive nature means every token serves as both input and target during training.
3. The uniform architecture (no separate encoder and decoder) simplifies training and optimization.

This scaling advantage is why the largest and most capable models today (GPT-4, Claude, Llama) are all decoder-only.

---

![Scaling laws and model architecture selection](https://picsum.photos/seed/encoder-vs-decoder-transformer-3/800/450)

### Part 8: Practical Guidance

**Choose encoder-only (BERT) when:**
- You need embeddings or representations for downstream tasks.
- Your task is classification, NER, or semantic similarity.
- Your dataset is small and you want to fine-tune efficiently.

**Choose decoder-only (GPT) when:**
- You need text generation.
- You want a general-purpose model.
- You are using prompt-based approaches (few-shot, zero-shot).
- You want to leverage the largest available models.

**Choose encoder-decoder (T5) when:**
- You have a clear input-output structure.
- Translation or summarization is your primary task.
- You want a structured approach to multiple NLP tasks.

---

### Final Thoughts

The encoder and decoder are not just architectural choices — they represent fundamentally different approaches to processing language. The encoder sees everything at once and builds understanding. The decoder generates one token at a time and builds output.

The remarkable finding of the last few years is that with sufficient scale, the decoder alone can approximate the capabilities of both. But understanding both halves — and why they were designed the way they were — is essential for making informed architectural decisions.

In the next post, we examine one of the Transformer's most underappreciated advantages: **How Transformers Parallelized Language and Changed Everything**.
