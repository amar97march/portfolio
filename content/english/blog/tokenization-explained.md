---
title: "What is Tokenization? Breaking Text into Pieces"
date: 2026-08-04T10:00:00+05:30
draft: false
description: "Tokenization is the first step in any NLP pipeline. Learn about word-level, character-level, and subword tokenization, and why modern LLMs use Byte Pair Encoding."
tags: ["NLP", "Tokenization", "Text Processing", "BPE", "Python"]
categories: ["NLP"]
image: "/images/blogs/pool-nlp/1.jpg"
keywords: ["tokenization NLP", "byte pair encoding", "BPE tokenizer", "wordpiece", "sentencepiece", "subword tokenization"]
---

Before a machine learning model can process text, that text must be broken into discrete units. This process is called **tokenization**, and it is the very first step in every NLP pipeline.

It sounds trivial — just split on spaces, right? In practice, tokenization is one of the most important design decisions in NLP, and the approach you choose has a profound impact on model performance.

---

### Part 1: Why Tokenization Matters

Consider these challenges:

- **"don't"** — Is this one token or two? If two, is it "don" + "'t" or "do" + "n't"?
- **"New York City"** — Should this be three tokens or one?
- **"unhappiness"** — Should the model see the whole word, or understand that "un-" means negation and "-ness" means a state?
- **"COVID-19"** — A new word that did not exist in any training corpus before 2020.
- **Chinese and Japanese text** — No spaces between words at all.
- **"lol"** vs. **"LOL"** vs. **"Lol"** — Same word? Different tokens?

The tokenizer determines what the model "sees." A bad tokenizer can cripple even the best model.

---

### Part 2: Word-Level Tokenization

The simplest approach: split text on spaces and punctuation.

```python
text = "The cat sat on the mat."
tokens = text.lower().split()
print(tokens)
# ['the', 'cat', 'sat', 'on', 'the', 'mat.']
```

**Advantages:**
- Simple and fast.
- Each token is a meaningful word.

**Problems:**
- **Huge vocabulary:** English has hundreds of thousands of words. Including all their forms (run, runs, running, ran) makes the vocabulary even larger.
- **Out-of-vocabulary (OOV) problem:** Any word not seen during training becomes an unknown token. Typos, new words, and rare terms are all lost.
- **No morphological awareness:** The model does not know that "running" is related to "run."

---

### Part 3: Character-Level Tokenization

At the opposite extreme: every character is a token.

```python
text = "Hello"
tokens = list(text)
print(tokens)
# ['H', 'e', 'l', 'l', 'o']
```

**Advantages:**
- **Tiny vocabulary:** Only ~100 unique characters (letters, digits, punctuation).
- **No OOV problem:** Every possible word can be represented.
- **Handles typos and new words** naturally.

**Problems:**
- **Very long sequences:** A 500-word document becomes 2,500+ tokens. This makes training slow and expensive.
- **Loss of meaning:** Individual characters carry almost no semantic information. The model must learn that "c-a-t" means "cat" — a much harder task.
- **Long-range dependencies:** Understanding that the "un" at the start of a word relates to the end of the word requires the model to look across many tokens.

---

![Comparing word-level and character-level tokenization approaches](/images/blogs/pool-nlp/3.jpg)

### Part 4: Subword Tokenization — The Sweet Spot

Modern NLP uses **subword tokenization** — a middle ground between word-level and character-level. Common words are kept whole, while rare words are broken into meaningful subparts.

For example:
- "running" might stay as "running" (common word)
- "unhappiness" might become "un" + "happiness" or "un" + "happi" + "ness"
- "transformerized" (rare word) might become "transform" + "er" + "ized"

This gives us the best of both worlds: a manageable vocabulary, no OOV problem, and morphological awareness.

---

### Part 5: Byte Pair Encoding (BPE)

BPE is the most widely used subword tokenization algorithm. It is used by GPT models, RoBERTa, and many other modern systems.

**How it works:**

1. **Start with characters.** Initialize the vocabulary with all individual characters in the training corpus.

2. **Count pairs.** Find the most frequent adjacent pair of tokens.

3. **Merge.** Replace all occurrences of that pair with a new token.

4. **Repeat.** Continue merging until the vocabulary reaches the desired size.

```python
# Simplified BPE demonstration
def simple_bpe(corpus, num_merges):
    """Simplified BPE to illustrate the concept."""
    # Start with character-level tokens
    vocab = {}
    for word in corpus:
        chars = ' '.join(list(word)) + ' </w>'
        vocab[chars] = vocab.get(chars, 0) + 1

    for i in range(num_merges):
        # Count all adjacent pairs
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pair = (symbols[j], symbols[j + 1])
                pairs[pair] = pairs.get(pair, 0) + freq

        if not pairs:
            break

        # Find the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        print(f"Merge {i+1}: '{best_pair[0]}' + '{best_pair[1]}' -> '{best_pair[0]}{best_pair[1]}'")

        # Merge the best pair in all words
        new_vocab = {}
        bigram = ' '.join(best_pair)
        replacement = ''.join(best_pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        vocab = new_vocab

    return vocab

# Example
corpus = ["low", "lower", "newest", "widest", "low", "low", "lower"]
result = simple_bpe(corpus, 10)
```

After training, the BPE tokenizer has a merge table that it applies to new text. Frequent words stay whole, and rare words are decomposed into known subword units.

---

### Part 6: WordPiece Tokenization

WordPiece is similar to BPE but uses a different criterion for merging. Instead of merging the most frequent pair, it merges the pair that **maximizes the likelihood of the training data**.

WordPiece is used by BERT and related models. It marks subword tokens that continue a word with "##":

```
"unbelievable" → ["un", "##believ", "##able"]
```

The "##" prefix indicates that this token is a continuation of the previous one, not a standalone word.

---

![Byte Pair Encoding merging tokens iteratively](/images/blogs/pool-nlp/4.jpg)

### Part 7: SentencePiece

SentencePiece, developed by Google, treats the input as a raw stream of characters (including spaces) rather than pre-tokenized words. This makes it **language-agnostic** — it works for Chinese, Japanese, and other languages where word boundaries are not marked by spaces.

SentencePiece supports both BPE and Unigram models.

```python
# Example with the tokenizers library (Hugging Face)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train on sample data
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=1000
)

# In practice, you would train on a large corpus
# tokenizer.train(files=["data.txt"], trainer=trainer)
```

---

### Part 8: Tokenization in Practice

Modern pre-trained models come with their own tokenizers. You should always use the tokenizer that matches your model:

```python
from transformers import AutoTokenizer

# Load BERT's tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love natural language processing!"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

# Decode back to text
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")
```

---

![Tokenization in practice with modern language models](/images/blogs/pool-nlp/5.jpg)

### Part 9: The Token Economy

Understanding tokenization is increasingly important because of the **token economy** in modern AI:

- **API pricing:** GPT-4 and Claude charge per token. Understanding how your text tokenizes helps you estimate costs.
- **Context window:** Models have maximum context lengths measured in tokens (4K, 32K, 128K, 1M). Knowing token count matters.
- **Generation speed:** Models generate one token at a time. More tokens = slower output.

A rough rule of thumb for English: **1 token is approximately 0.75 words** (or 4 characters). So a 1,000-word document is roughly 1,300 tokens.

---

### Part 10: Choosing a Tokenization Strategy

| Strategy | Vocabulary Size | OOV Handling | Semantic Quality | Speed |
|----------|----------------|-------------|-----------------|-------|
| **Word-level** | Very large (100K+) | Poor | High per token | Fast |
| **Character-level** | Very small (~100) | Perfect | Very low per token | Slow (long sequences) |
| **BPE** | Configurable (30K-50K typical) | Good | Good | Fast |
| **WordPiece** | Configurable | Good | Good | Fast |
| **SentencePiece** | Configurable | Good | Good | Fast |

For new projects using pre-trained models, use the model's bundled tokenizer. For custom models, BPE with a vocabulary of 30,000-50,000 tokens is a strong default.

---

### Final Thoughts

Tokenization is the bridge between human language and machine computation. The choice of tokenizer determines what information the model can access, how efficiently it processes text, and ultimately how well it performs.

Modern subword tokenizers like BPE represent a elegant solution to the vocabulary problem — balancing coverage, efficiency, and morphological awareness.

In the next post, we go deeper into representation: **What are Embeddings? Turning Words into Numbers**.
