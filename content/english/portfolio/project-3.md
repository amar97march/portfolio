---
title: "RAG Knowledge Assistant"
date: 2025-03-10T09:00:00+05:30
draft: false
description: "Retrieval-Augmented Generation system that surfaces domain knowledge from internal documents, reducing response times for operators at Antro-PAI."
tags: ["Python", "FastAPI", "NLP", "RAG", "LLM"]
categories: ["ML/AI"]
image: "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&h=500&fit=crop"
company: "Antro-PAI"
role: "ML Engineer"
year: "2024-2025"
tech_stack: ["Python", "FastAPI", "LangChain", "ChromaDB", "OpenAI API", "Docker"]
category: "ml-ai"
featured: true
---

Designed and deployed a Retrieval-Augmented Generation (RAG) pipeline at Antro-PAI that enables operators to query internal knowledge bases using natural language.

## Challenge

Operators needed quick access to scattered domain knowledge across hundreds of internal documents. Manual search was slow and often returned irrelevant results.

## Solution

Built a RAG pipeline that ingests, chunks, and indexes internal documents into a vector store. User queries are embedded and matched against the knowledge base, with relevant context fed to an LLM for grounded, accurate responses.

## Key Contributions

- Designed the document ingestion pipeline with smart chunking strategies
- Built FastAPI service with streaming response support
- Implemented hybrid search combining semantic similarity and keyword matching
- Set up evaluation framework to measure retrieval accuracy and response quality
- Deployed with Docker on cloud infrastructure

## Impact

- Reduced average query resolution time significantly
- Improved answer accuracy by grounding LLM responses in verified internal documents
