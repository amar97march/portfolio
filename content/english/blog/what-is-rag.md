---
title: "What is RAG? Retrieval-Augmented Generation Explained"
date: 2026-10-24T10:00:00+05:30
draft: false
description: "A comprehensive guide to Retrieval-Augmented Generation — the technique that grounds LLM responses in real data to reduce hallucination and enable knowledge-based AI applications."
tags: ["RAG", "LLM", "Vector Database", "NLP", "Generative AI"]
categories: ["RAG & Retrieval"]
image: "https://picsum.photos/seed/what-is-rag-cover/1200/630"
keywords: ["RAG explained", "retrieval augmented generation", "RAG LLM", "grounded AI", "knowledge retrieval AI", "RAG architecture"]
---

If hallucination is the biggest problem with LLMs, then **RAG (Retrieval-Augmented Generation)** is the most practical solution. RAG combines the generative power of language models with the factual grounding of information retrieval, producing responses that are both fluent and accurate.

RAG is not just a research concept — it is the architecture behind most production LLM applications today. Every time you use an AI chatbot that references company documentation, a search engine that generates AI summaries, or a code assistant that understands your specific codebase, you are likely interacting with a RAG system.

### The Core Idea

RAG is deceptively simple in concept:

1. **Retrieve** relevant documents based on the user's query
2. **Augment** the LLM's prompt with those retrieved documents
3. **Generate** a response grounded in the retrieved information

Instead of asking the LLM to answer from memory (its parametric knowledge), you give it an open-book exam with the relevant information right in front of it.

```
Without RAG:
User → LLM → Response (from memory, might hallucinate)

With RAG:
User → Retriever → Relevant Documents → LLM → Response (grounded in docs)
```

### Why RAG Matters

**Problem 1: Knowledge cutoff**
LLMs only know what was in their training data. RAG lets you provide up-to-date information.

**Problem 2: Hallucination**
LLMs can generate false information confidently. RAG grounds responses in actual documents.

**Problem 3: Domain knowledge**
Your company's internal documentation, codebase, and policies were not in the training data. RAG gives the LLM access to this proprietary knowledge.

**Problem 4: Verifiability**
With RAG, responses can cite specific source documents. Users can verify claims by checking the sources.

### The RAG Architecture

![RAG architecture showing indexing and querying phases](https://picsum.photos/seed/what-is-rag-1/800/450)


A complete RAG system has two main phases: **indexing** (done once or periodically) and **querying** (done at runtime).

#### Indexing Phase

```
Documents → Chunking → Embedding → Vector Database

1. Load documents (PDFs, web pages, code, databases)
2. Split into chunks (paragraphs, sections, or fixed-size windows)
3. Generate embeddings for each chunk (convert text to vectors)
4. Store embeddings in a vector database for fast retrieval
```

#### Querying Phase

```
User Query → Embedding → Vector Search → Top-K Chunks → LLM → Response

1. Convert the user's query into an embedding
2. Search the vector database for similar chunks
3. Retrieve the top-K most relevant chunks
4. Combine query + retrieved chunks into an LLM prompt
5. Generate response grounded in the retrieved context
```

### A Minimal RAG Implementation

Let us build a basic RAG system step by step:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# Step 1: Your knowledge base (in practice, loaded from files)
documents = [
    "Our refund policy allows returns within 30 days of purchase. "
    "Items must be in original packaging.",
    "Premium members get free shipping on all orders over $25. "
    "Standard shipping is $5.99.",
    "Our support hours are Monday through Friday, 9 AM to 6 PM EST. "
    "Weekend support is available via email only.",
    "We accept Visa, Mastercard, and PayPal. "
    "We do not accept cryptocurrency payments.",
]

# Step 2: Generate embeddings for all documents
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

doc_embeddings = [get_embedding(doc) for doc in documents]

# Step 3: Retrieve relevant documents
def retrieve(query, top_k=2):
    query_embedding = get_embedding(query)

    # Calculate cosine similarity
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        similarity = np.dot(query_embedding, doc_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
        )
        similarities.append((similarity, i))

    # Return top-k most similar documents
    similarities.sort(reverse=True)
    return [documents[i] for _, i in similarities[:top_k]]

# Step 4: Generate response with retrieved context
def rag_query(user_question):
    relevant_docs = retrieve(user_question)

    context = "\n\n".join(relevant_docs)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Answer the user's question based ONLY on "
                           "the provided context. If the context does "
                           "not contain the answer, say 'I don't have "
                           "information about that in my knowledge base.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n"
                           f"Question: {user_question}"
            }
        ]
    )
    return response.choices[0].message.content

# Usage
answer = rag_query("Can I return an item after 2 weeks?")
print(answer)
# "Yes, you can return items within 30 days of purchase,
#  so 2 weeks would be within the return window. The item
#  must be in its original packaging."
```

This is a complete, working RAG system in about 50 lines. Production systems add more sophistication, but the core pattern is exactly this.

### Key Components Deep Dive

![Vector similarity search retrieving relevant documents](https://picsum.photos/seed/what-is-rag-2/800/450)


#### Embeddings

An embedding is a vector (list of numbers) that represents the semantic meaning of text. Texts with similar meanings have similar embeddings, even if they use different words.

```
"How do I get a refund?" → [0.12, -0.34, 0.78, ...]
"I want to return my purchase" → [0.11, -0.32, 0.75, ...]
"What's the weather today?" → [-0.45, 0.22, -0.18, ...]
```

The first two vectors are similar (both about returns/refunds). The third is very different (about weather). This is what enables semantic search — finding relevant documents by meaning rather than keyword matching.

**Popular embedding models:**
- OpenAI `text-embedding-3-small` / `text-embedding-3-large`
- Cohere `embed-v3`
- Open-source: `BAAI/bge-large`, `sentence-transformers`

#### Chunking

Documents must be split into smaller pieces because:
1. LLMs have context window limits
2. Smaller chunks enable more precise retrieval
3. Large documents dilute the relevant information

**Common chunking strategies:**

```python
# Fixed-size chunking
def fixed_chunk(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Semantic chunking (split on natural boundaries)
def semantic_chunk(text):
    # Split on paragraph breaks, section headers, etc.
    paragraphs = text.split('\n\n')
    return [p for p in paragraphs if len(p) > 50]
```

The overlap between chunks ensures that information at chunk boundaries is not lost.

#### Vector Search

Finding the most similar embeddings in a large collection. For a few thousand documents, brute-force cosine similarity works. For millions, you need approximate nearest neighbor (ANN) algorithms:

- **HNSW** (Hierarchical Navigable Small World) — most popular
- **IVF** (Inverted File Index) — good for large datasets
- **Product Quantization** — memory-efficient for massive collections

This is where vector databases come in — they implement these algorithms efficiently.

### RAG vs. Fine-Tuning

A common question: should I use RAG or fine-tune the model?

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge updates | Easy (re-index documents) | Hard (re-train model) |
| Source citation | Yes (can point to documents) | No |
| Hallucination reduction | Strong | Moderate |
| Cost | Low (no training required) | High (GPU time) |
| Custom behavior | Limited (same model behavior) | Strong (change model style) |
| Latency | Higher (retrieval step adds time) | Lower (single model call) |

**Use RAG when:** You need to give the model access to specific, up-to-date knowledge.
**Use fine-tuning when:** You need to change how the model behaves, not what it knows.
**Use both when:** You need custom behavior AND specific knowledge.

### Common RAG Pitfalls

![Comparison of RAG versus fine-tuning approaches](https://picsum.photos/seed/what-is-rag-3/800/450)


**1. Poor chunking:**
If chunks are too large, retrieval is imprecise. If too small, chunks lack context. Experiment with 200-500 token chunks with 50-100 token overlap as a starting point.

**2. Irrelevant retrieval:**
The retrieved documents might not actually answer the question. Always include an instruction to the LLM: "If the context doesn't contain the answer, say so."

**3. Context window overflow:**
Retrieving too many documents can overflow the context window or dilute relevant information. Start with top-3 retrieval and increase only if needed.

**4. Embedding mismatch:**
Use the same embedding model for indexing and querying. Mixing models produces poor similarity scores.

### Where RAG Is Heading

The field is evolving rapidly:

- **Hybrid search**: Combining vector search with traditional keyword search (BM25) for better retrieval
- **Re-ranking**: Using a separate model to re-rank retrieved documents before passing them to the LLM
- **Agentic RAG**: LLMs that decide when and what to retrieve, rather than always retrieving
- **Multi-modal RAG**: Retrieving images, tables, and code alongside text

RAG is the bridge between the general intelligence of LLMs and the specific knowledge your application needs. In the next posts, we will explore the RAG workflow in detail, vector databases, and how to build a production-quality "Chat With Your Docs" application.
