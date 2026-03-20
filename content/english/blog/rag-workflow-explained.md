---
title: "The RAG Workflow: Search, Retrieve, Augment, Generate"
date: 2026-10-27T10:00:00+05:30
draft: false
description: "A step-by-step walkthrough of the complete RAG pipeline — from document ingestion to response generation — with code examples and optimization tips."
tags: ["RAG", "LLM", "Vector Search", "AI Pipeline", "Generative AI"]
categories: ["RAG & Retrieval"]
image: "/images/blogs/pool-rag/1.jpg"
keywords: ["RAG workflow", "RAG pipeline", "document retrieval", "semantic search", "RAG implementation", "AI pipeline architecture"]
---

In the previous post, we introduced RAG at a high level. Now let us get our hands dirty and walk through the complete RAG workflow step by step. By the end of this post, you will understand every stage of the pipeline and how to optimize each one.

### The Four Stages

A RAG workflow has four distinct stages, each with its own engineering challenges:

1. **Ingest**: Load, clean, and chunk your documents
2. **Index**: Generate embeddings and store them in a vector database
3. **Retrieve**: Find the most relevant chunks for a given query
4. **Generate**: Combine retrieved context with the query and produce a response

Let us dive into each stage.

### Stage 1: Ingest — Preparing Your Documents

The quality of your RAG system starts with the quality of your document processing. Garbage in, garbage out.

**Loading documents from various formats:**

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)

def load_documents(file_path: str):
    """Load documents based on file extension."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    else:
        loader = TextLoader(file_path)

    return loader.load()
```

**Cleaning the text:**

Raw documents contain noise — headers, footers, navigation elements, formatting artifacts. Clean them:

```python
import re

def clean_text(text: str) -> str:
    """Clean extracted text for better embedding quality."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and headers
    text = re.sub(r'Page \d+ of \d+', '', text)
    # Remove URLs (optional, depends on use case)
    # text = re.sub(r'https?://\S+', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text
```

**Chunking — the most critical step:**

How you split documents into chunks has the single biggest impact on retrieval quality.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)

    return chunks
```

**Chunking strategies compared:**


![Illustration of knowledge base integration with language models](/images/blogs/pool-rag/5.jpg)

| Strategy | Chunk Size | Overlap | Best For |
|----------|-----------|---------|----------|
| Small chunks | 200-300 chars | 50 | Precise Q&A |
| Medium chunks | 500-800 chars | 100 | General RAG |
| Large chunks | 1000-1500 chars | 200 | Complex reasoning |
| Semantic chunks | Variable | N/A | Well-structured docs |

The right choice depends on your use case. Start with medium chunks and adjust based on retrieval quality.

### Stage 2: Index — Embedding and Storing

Once you have chunks, convert them to embeddings and store them for fast retrieval.

```python
from openai import OpenAI
import chromadb

client = OpenAI()

def create_index(chunks):
    """Create a vector index from document chunks."""

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )

    # Process chunks in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Generate embeddings
        texts = [chunk.page_content for chunk in batch]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [r.embedding for r in response.data]

        # Store in ChromaDB
        collection.add(
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[chunk.metadata for chunk in batch]
        )

    return collection
```

**Choosing an embedding model:**

The embedding model determines the quality of semantic similarity matching:

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| text-embedding-3-small | 1536 | Fast | Good | $0.02/1M tokens |
| text-embedding-3-large | 3072 | Medium | Better | $0.13/1M tokens |
| BGE-large-en | 1024 | Fast | Good | Free (open source) |
| Cohere embed-v3 | 1024 | Fast | Excellent | $0.10/1M tokens |

For most applications, `text-embedding-3-small` provides excellent quality at low cost.

### Stage 3: Retrieve — Finding Relevant Context

Retrieval is where the magic happens. Given a user query, find the most relevant chunks.

```python
def retrieve(collection, query: str, top_k: int = 5):
    """Retrieve the most relevant chunks for a query."""

    # Generate query embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    # Search vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    return results
```

**Improving retrieval quality:**


![Visual representation of vector search and document retrieval](/images/blogs/pool-rag/4.jpg)

**Hybrid search** combines vector search with keyword search:

```python
def hybrid_retrieve(collection, query, top_k=5, alpha=0.7):
    """Combine semantic search with keyword search."""

    # Semantic search (vector similarity)
    semantic_results = vector_search(collection, query, top_k * 2)

    # Keyword search (BM25 or similar)
    keyword_results = bm25_search(collection, query, top_k * 2)

    # Combine scores
    combined = {}
    for doc_id, score in semantic_results:
        combined[doc_id] = alpha * score
    for doc_id, score in keyword_results:
        combined[doc_id] = combined.get(doc_id, 0) + (1 - alpha) * score

    # Return top-k by combined score
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
```

**Re-ranking** uses a cross-encoder model to re-score retrieved documents:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, top_k=3):
    """Re-rank retrieved documents using a cross-encoder."""
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

Re-ranking is computationally more expensive than vector search but significantly improves precision.

### Stage 4: Generate — Producing the Response

The final stage combines retrieved context with the user's query and generates a response.

```python
def generate_response(query: str, retrieved_docs: list) -> str:
    """Generate a response using retrieved context."""

    # Format context
    context = "\n\n---\n\n".join([
        f"Source {i+1}:\n{doc}"
        for i, doc in enumerate(retrieved_docs)
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant that answers
                questions based on provided context. Follow these rules:
                1. Base your answer ONLY on the provided context
                2. If the context doesn't contain the answer, say so
                3. Cite which source(s) your answer comes from
                4. Be concise but thorough
                5. If the context is ambiguous, explain the ambiguity"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n"
                           f"Question: {query}\n\n"
                           f"Answer (cite your sources):"
            }
        ],
        temperature=0.1  # Low temperature for factual responses
    )

    return response.choices[0].message.content
```

### The Complete Pipeline

Putting it all together:


![Diagram of retrieval-augmented generation architecture](/images/blogs/pool-rag/3.jpg)

```python
class RAGPipeline:
    def __init__(self, documents_dir: str):
        self.client = OpenAI()
        self.collection = None
        self.index_documents(documents_dir)

    def index_documents(self, documents_dir):
        """Ingest, chunk, embed, and index all documents."""
        docs = load_all_documents(documents_dir)
        chunks = chunk_documents(docs)
        self.collection = create_index(chunks)
        print(f"Indexed {len(chunks)} chunks from {len(docs)} documents")

    def query(self, user_question: str) -> str:
        """Full RAG pipeline: retrieve and generate."""
        # Retrieve
        results = retrieve(self.collection, user_question, top_k=5)
        documents = results['documents'][0]

        # Optional: re-rank
        documents = rerank(user_question, documents, top_k=3)

        # Generate
        response = generate_response(user_question, documents)
        return response

# Usage
rag = RAGPipeline("./company_docs/")
answer = rag.query("What is our vacation policy for remote employees?")
```

### Optimizing Each Stage

**Ingest optimizations:**
- Use metadata-aware chunking (preserve section headers, page numbers)
- Add parent-child relationships between chunks for context expansion
- Pre-process tables and structured data separately from prose

**Index optimizations:**
- Batch embedding requests to reduce API calls
- Use dimensionality reduction for large collections
- Create separate collections for different document types

**Retrieval optimizations:**
- Implement query expansion (generate multiple search queries from one user question)
- Use MMR (Maximum Marginal Relevance) to ensure diversity in retrieved results
- Filter by metadata (date, source, category) before vector search

**Generation optimizations:**
- Include a system prompt that matches your use case
- Use structured output formats for consistency
- Implement streaming for better user experience

### Evaluation: How Do You Know It Works?

RAG evaluation is critical. Measure these metrics:

```python
def evaluate_rag(test_set):
    """Evaluate RAG system on a test set."""
    metrics = {
        'retrieval_recall': [],     # Did we retrieve the right docs?
        'answer_relevancy': [],      # Is the answer relevant?
        'faithfulness': [],          # Is the answer grounded in context?
        'answer_correctness': [],    # Is the answer correct?
    }

    for question, expected_answer, expected_sources in test_set:
        retrieved = retrieve(question)
        answer = generate(question, retrieved)

        metrics['retrieval_recall'].append(
            check_source_overlap(retrieved, expected_sources)
        )
        metrics['answer_relevancy'].append(
            score_relevancy(question, answer)
        )
        metrics['faithfulness'].append(
            check_grounding(answer, retrieved)
        )

    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

Tools like **Ragas**, **TruLens**, and **LangSmith** provide automated evaluation frameworks for RAG systems.

### The RAG workflow is the backbone of modern AI applications. Master each stage — ingest, index, retrieve, generate — and you can build systems that combine the intelligence of LLMs with the accuracy of your specific knowledge base. In the next posts, we will explore vector databases in depth and build a complete "Chat With Your Docs" application.
