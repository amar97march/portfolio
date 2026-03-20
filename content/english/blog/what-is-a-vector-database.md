---
title: "What is a Vector Database? The Brain Behind RAG"
date: 2026-10-30T10:00:00+05:30
draft: false
description: "A complete guide to vector databases — how they work, why they matter for AI applications, and how to choose the right one for your RAG pipeline."
tags: ["Vector Database", "RAG", "Embeddings", "AI Infrastructure", "Generative AI"]
categories: ["RAG & Retrieval"]
image: "/images/blogs/pool-rag/1.jpg"
keywords: ["vector database", "vector DB", "embedding database", "similarity search", "Pinecone", "Chroma", "Milvus", "HNSW"]
---

Behind every RAG system, there is a vector database doing the heavy lifting of semantic search. It is the component that takes a question like "How do I reset my password?" and finds documents about "account recovery procedures" — even though the two phrases share no keywords.

Vector databases are one of the fastest-growing categories in the database world, and understanding them is essential for building any modern AI application.

### What Makes Vector Databases Different

Traditional databases are designed around structured data — rows, columns, indexes on exact values. You query them with exact matches, ranges, and joins:

```sql
-- Traditional query: exact match
SELECT * FROM products WHERE category = 'electronics' AND price < 100;
```

Vector databases are designed around **similarity search** — finding the items most similar to a given vector. You query them by distance:

```python
# Vector query: find similar items
results = vector_db.search(
    query_vector=[0.12, -0.34, 0.78, ...],  # 1536 dimensions
    top_k=5,                                  # Return 5 most similar
    metric="cosine"                           # Similarity metric
)
```

The vectors represent semantic meaning. Two documents about the same topic will have similar vectors, even if they use completely different words.

### How Vector Databases Work

#### Step 1: Storing Vectors

When you add a document to a vector database, you store:
- The **embedding vector** (e.g., 1536 floating-point numbers)
- The **original content** (the text, or a reference to it)
- **Metadata** (source, date, category, etc.)

```python
# Storing a document in a vector database
collection.add(
    ids=["doc_001"],
    embeddings=[[0.12, -0.34, 0.78, ...]],  # 1536-dim vector
    documents=["Our return policy allows returns within 30 days"],
    metadatas=[{"source": "policy.pdf", "page": 5}]
)
```

#### Step 2: Indexing for Fast Search

The naive approach to similarity search is to compare the query vector against every stored vector. This is O(n) and becomes impossibly slow at scale.

Vector databases use **Approximate Nearest Neighbor (ANN)** algorithms to make search sub-linear:

**HNSW (Hierarchical Navigable Small World):**
The most popular algorithm. It builds a multi-layer graph where each node is connected to its nearest neighbors. Search starts at the top layer (sparse, long-range connections) and navigates down to the bottom layer (dense, short-range connections).

```
Layer 3: [A] ---- [D] ---- [H]        (few nodes, long jumps)
Layer 2: [A] - [C] - [D] - [F] - [H]  (more nodes)
Layer 1: [A][B][C][D][E][F][G][H]      (all nodes, short jumps)

Search: Start at top → navigate to approximate area → refine at bottom
```

HNSW provides:
- Search in O(log n) time
- High recall (95-99% accuracy)
- Good performance for millions of vectors

**IVF (Inverted File Index):**
Clusters vectors into groups. At query time, only searches the nearest clusters:

```
Cluster 1: [doc_1, doc_4, doc_7, ...]
Cluster 2: [doc_2, doc_5, doc_8, ...]
Cluster 3: [doc_3, doc_6, doc_9, ...]

Query → Find nearest cluster → Search within that cluster
```

#### Step 3: Similarity Search

At query time, the database:
1. Takes the query vector
2. Navigates the index to find approximate nearest neighbors
3. Computes exact similarity for the candidates
4. Returns the top-k results

Common similarity metrics:

```python
import numpy as np

def cosine_similarity(a, b):
    """Most popular for text embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """Good for image embeddings."""
    return np.sqrt(np.sum((a - b) ** 2))

def dot_product(a, b):
    """Fast, works when vectors are normalized."""
    return np.dot(a, b)
```

Cosine similarity is the standard choice for text embeddings because it measures the angle between vectors regardless of magnitude.

![How vector databases index and search embeddings](/images/blogs/pool-rag/3.jpg)

### Key Features of Vector Databases

**Metadata filtering:**
Filter results by metadata before or after vector search:

```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={
        "source": "technical_docs",
        "date": {"$gte": "2024-01-01"}
    }
)
```

**Namespaces/Collections:**
Organize vectors into logical groups:

```python
# Separate collections for different document types
technical_docs = db.create_collection("technical_docs")
policy_docs = db.create_collection("policy_docs")
support_tickets = db.create_collection("support_tickets")
```

**Hybrid search:**
Combine vector similarity with keyword search:

```python
# Some databases support hybrid queries natively
results = collection.hybrid_search(
    query_vector=embedding,    # Semantic search
    query_text="reset password", # Keyword search
    alpha=0.7                    # Weight toward semantic
)
```

### Choosing a Vector Database

The ecosystem is growing rapidly. Here is how the major options compare:

**Managed cloud services:**
- **Pinecone**: Fully managed, easy to use, good for production
- **Weaviate Cloud**: Feature-rich, supports hybrid search natively
- **Qdrant Cloud**: High performance, good filtering

**Self-hosted options:**
- **Milvus**: Mature, highly scalable, GPU-accelerated
- **Qdrant**: Rust-based, fast, good developer experience
- **Weaviate**: Go-based, strong hybrid search

**Lightweight/embedded:**
- **ChromaDB**: Python-native, great for prototyping
- **LanceDB**: Embedded, serverless, built on Lance format
- **SQLite-VSS**: Vector search extension for SQLite

**When to use which:**

| Scenario | Recommendation |
|----------|---------------|
| Prototyping / learning | ChromaDB |
| Small production app (<100K docs) | Qdrant or ChromaDB |
| Large scale (millions of docs) | Pinecone or Milvus |
| Need hybrid search | Weaviate |
| Need GPU acceleration | Milvus |
| Want serverless | Pinecone or LanceDB |
| Want to avoid vendor lock-in | Qdrant or Milvus (self-hosted) |

![Choosing the right vector database for your AI application](/images/blogs/pool-rag/4.jpg)

### Practical Example: Building a Searchable Knowledge Base

```python
import chromadb
from openai import OpenAI

# Initialize
openai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./my_knowledge_base")

# Create collection
collection = chroma_client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}
)

def add_document(doc_id: str, text: str, metadata: dict):
    """Add a document to the knowledge base."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )

def search(query: str, n_results: int = 5, filter_metadata: dict = None):
    """Search the knowledge base."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    search_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if filter_metadata:
        search_params["where"] = filter_metadata

    results = collection.query(**search_params)
    return results

# Add documents
add_document(
    "policy-001",
    "Employees are entitled to 20 days of paid leave per year.",
    {"category": "hr", "type": "policy"}
)

add_document(
    "tech-001",
    "All API endpoints must use JWT authentication.",
    {"category": "engineering", "type": "standard"}
)

# Search
results = search("How many vacation days do I get?")
# Returns the HR policy document
```

### Performance Considerations

**Embedding dimensions:**
Higher dimensions capture more semantic nuance but require more storage and slower search. The sweet spot for most applications is 768-1536 dimensions.

**Index build time vs. query speed:**
HNSW parameters control this trade-off:
- `M` (connections per node): Higher = better recall, more memory
- `ef_construction`: Higher = better index quality, slower build
- `ef_search`: Higher = better recall at query time, slower search

```python
# ChromaDB HNSW configuration
collection = client.create_collection(
    name="optimized_docs",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 32,                 # Default: 16
        "hnsw:construction_ef": 200,   # Default: 100
        "hnsw:search_ef": 100          # Default: 10
    }
)
```

**Memory usage:**
A single 1536-dimension float32 vector uses ~6KB. One million vectors use ~6GB. Plan your infrastructure accordingly.

![Vector databases as a critical component of the modern AI stack](/images/blogs/pool-rag/5.jpg)

### Vector Databases in the AI Stack

Vector databases are not standalone tools — they are a critical component of the modern AI application stack:

```
User Query
    ↓
Application Layer (your code)
    ↓
Vector Database (semantic search)  ←→  Embedding Model (vectorization)
    ↓
Retrieved Context
    ↓
LLM (response generation)
    ↓
Response to User
```

They sit between the user's intent and the LLM's generation, ensuring that the model has access to relevant, specific information rather than relying on its general training.

Understanding vector databases is essential for any developer building AI-powered applications. In the next post, we will do a hands-on comparison of the three most popular vector databases: Pinecone, Milvus, and Chroma.
