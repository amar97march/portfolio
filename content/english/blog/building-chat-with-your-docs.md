---
title: "Use Case: Building a Chat With Your Docs Application"
date: 2026-11-05T10:00:00+05:30
draft: false
description: "A hands-on tutorial for building a production-quality 'Chat With Your Docs' application using RAG, vector databases, and LLMs."
tags: ["RAG", "Tutorial", "LLM", "ChromaDB", "Python", "Project"]
categories: ["RAG & Retrieval"]
image: "https://images.unsplash.com/photo-1623282033815-40b8d0960e4b?w=1200&h=630&fit=crop&auto=format"
keywords: ["chat with documents", "RAG tutorial", "document chatbot", "build RAG app", "LLM document search", "AI document assistant"]
---

We have covered the theory — RAG architecture, vector databases, retrieval strategies. Now let us build something real. In this post, we will build a complete **"Chat With Your Docs"** application from scratch — the kind of tool that lets you upload documents and ask questions about them in natural language.

This is one of the most requested AI applications, and for good reason. Every organization has documentation that people struggle to navigate. A well-built document chatbot can save hours of searching.

### What We Are Building

A Python application that:
1. Accepts PDF, Markdown, and text documents
2. Processes and indexes them in a vector database
3. Answers questions based on the document content
4. Cites sources so users can verify answers
5. Handles conversational context (follow-up questions)

### The Architecture

```
User Question
    ↓
Query Rewriting (handle follow-ups)
    ↓
Embedding Generation (OpenAI)
    ↓
Vector Search (ChromaDB)
    ↓
Re-ranking (optional)
    ↓
Context Assembly
    ↓
LLM Generation (GPT-4o)
    ↓
Response with Citations
```


![Diagram showing the retrieval-augmented generation architecture](https://picsum.photos/seed/building-chat-with-your-docs-1/800/450)

### Step 1: Project Setup

```bash
mkdir doc-chat && cd doc-chat
python -m venv venv
source venv/bin/activate
pip install openai chromadb langchain pypdf python-dotenv
```

```python
# .env
OPENAI_API_KEY=your-key-here
```

### Step 2: Document Ingestion

```python
# ingest.py
import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

LOADERS = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.md': UnstructuredMarkdownLoader,
}

def load_documents(directory: str) -> list:
    """Load all supported documents from a directory."""
    documents = []
    doc_dir = Path(directory)

    for file_path in doc_dir.rglob('*'):
        ext = file_path.suffix.lower()
        if ext in LOADERS:
            try:
                loader = LOADERS[ext](str(file_path))
                docs = loader.load()
                # Add source filename to metadata
                for doc in docs:
                    doc.metadata['source'] = file_path.name
                documents.extend(docs)
                print(f"Loaded: {file_path.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

    return documents

def chunk_documents(documents: list, chunk_size=500, overlap=100) -> list:
    """Split documents into chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"chunk_{i}"

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
```

### Step 3: Vector Store

```python
# vector_store.py
import chromadb
from openai import OpenAI

client = OpenAI()

def get_collection():
    """Get or create the ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def index_chunks(chunks: list):
    """Index document chunks in the vector store."""
    collection = get_collection()
    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.page_content for c in batch]
        embeddings = generate_embeddings(texts)

        collection.upsert(
            ids=[c.metadata['chunk_id'] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.metadata for c in batch],
        )
        print(f"Indexed batch {i // batch_size + 1}")

    print(f"Indexing complete: {len(chunks)} chunks")

def search(query: str, top_k: int = 5, filter_source: str = None):
    """Search for relevant chunks."""
    collection = get_collection()
    query_embedding = generate_embeddings([query])[0]

    params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if filter_source:
        params["where"] = {"source": filter_source}

    return collection.query(**params)
```


![Visual illustrating document processing and vector search workflow](https://picsum.photos/seed/building-chat-with-your-docs-2/800/450)

### Step 4: The Chat Engine

```python
# chat.py
from openai import OpenAI
from vector_store import search

client = OpenAI()

SYSTEM_PROMPT = """You are a helpful document assistant. You answer
questions based ONLY on the provided context from the user's documents.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say: "I couldn't find
   this information in your documents."
3. Always cite your sources using [Source: filename] format
4. Be concise but thorough
5. If the question is ambiguous, ask for clarification
6. Never make up information not present in the context"""

class DocChat:
    def __init__(self):
        self.conversation_history = []

    def _rewrite_query(self, user_message: str) -> str:
        """Rewrite follow-up questions to be standalone."""
        if not self.conversation_history:
            return user_message

        # Use the LLM to make the query standalone
        rewrite_messages = [
            {
                "role": "system",
                "content": "Given a conversation history and a follow-up "
                           "question, rewrite the question to be standalone "
                           "(understandable without the conversation history). "
                           "Return ONLY the rewritten question."
            },
            {
                "role": "user",
                "content": f"Conversation:\n"
                           f"{self._format_history()}\n\n"
                           f"Follow-up: {user_message}\n\n"
                           f"Standalone question:"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=rewrite_messages,
            temperature=0.0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    def _format_history(self) -> str:
        """Format conversation history for context."""
        formatted = []
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content'][:200]}")
        return "\n".join(formatted)

    def _build_context(self, results) -> str:
        """Build context string from search results."""
        context_parts = []
        for i, (doc, metadata) in enumerate(
            zip(results['documents'][0], results['metadatas'][0])
        ):
            source = metadata.get('source', 'Unknown')
            context_parts.append(
                f"[Source: {source}]\n{doc}"
            )
        return "\n\n---\n\n".join(context_parts)

    def ask(self, user_message: str) -> str:
        """Process a user question and return a response."""
        # Step 1: Rewrite query for standalone context
        standalone_query = self._rewrite_query(user_message)

        # Step 2: Retrieve relevant documents
        results = search(standalone_query, top_k=5)

        # Step 3: Build context
        context = self._build_context(results)

        # Step 4: Generate response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context from documents:\n{context}\n\n"
                           f"Question: {user_message}"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
        )

        answer = response.choices[0].message.content

        # Step 5: Update conversation history
        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": answer}
        )

        return answer
```

### Step 5: Putting It All Together

```python
# main.py
from ingest import load_documents, chunk_documents
from vector_store import index_chunks
from chat import DocChat

def setup(docs_directory: str):
    """Index documents from a directory."""
    print("Loading documents...")
    documents = load_documents(docs_directory)

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print("Indexing chunks...")
    index_chunks(chunks)

    print("Setup complete!")

def chat():
    """Start an interactive chat session."""
    doc_chat = DocChat()
    print("Document Chat Ready! Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if not question:
            continue

        answer = doc_chat.ask(question)
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        docs_dir = sys.argv[2] if len(sys.argv) > 2 else "./documents"
        setup(docs_dir)
    else:
        chat()
```

**Usage:**

```bash
# Index your documents
python main.py index ./my_documents/

# Start chatting
python main.py
```


![Conceptual image of an AI system grounding its responses in source documents](https://picsum.photos/seed/building-chat-with-your-docs-3/800/450)

### Key Design Decisions Explained

**Why query rewriting?**
Without it, follow-up questions lose context. If the user asks "What is the refund policy?" followed by "How long do I have?", the second query alone is not searchable. Rewriting turns it into "How long do I have to request a refund under the refund policy?"

**Why low temperature (0.1)?**
For factual Q&A over documents, you want deterministic, consistent responses. Higher temperatures introduce creativity — which in this context means making things up.

**Why chunk overlap?**
Information often spans chunk boundaries. A 100-token overlap ensures that sentences split across chunks are captured in at least one chunk fully.

**Why cite sources?**
Trust. Users need to verify that the AI is not hallucinating. Source citations let them check the original document.

### Production Improvements

For a production deployment, consider adding:

1. **Authentication**: Protect the API with user authentication
2. **Document management**: Upload, delete, and update documents through an API
3. **Streaming responses**: Use SSE to stream the LLM response for better UX
4. **Evaluation pipeline**: Regularly test with a set of known Q&A pairs
5. **Monitoring**: Track retrieval quality, response latency, and user satisfaction
6. **Rate limiting**: Prevent abuse and manage API costs
7. **Caching**: Cache frequent queries to reduce latency and cost

### Lessons From Production

After building several document chat applications for production use, here are my key lessons:

1. **Chunking quality is everything.** Spend 80% of your optimization effort on how you chunk documents. Bad chunks make everything downstream worse.

2. **Users ask vague questions.** Build robust query rewriting and handle ambiguity gracefully.

3. **"I don't know" is a feature.** Users trust a system that admits when it does not have the answer more than one that always gives an answer (even if wrong).

4. **Source citations are non-negotiable.** Every production system I have built requires citations. Users will not trust a system that does not show its sources.

5. **Start simple.** This basic architecture handles 90% of use cases. Add complexity (re-ranking, hybrid search, agentic retrieval) only when you have evidence that the simple approach is insufficient.

This is the foundation. In the next post, we will compare RAG and fine-tuning — two complementary approaches for building knowledge-specific AI applications.
