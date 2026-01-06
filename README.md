# rag-system
A Retrieval-Augmented Generation (RAG) system implemented using a custom vector database, Hugging Face embeddings, and GPT-2 for context-aware text generation.
# Retrieval-Augmented Generation (RAG) System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using a custom vector database and Hugging Face models.

## Components
- Custom vector database with cosine similarity search
- Embedding model: BGE-micro
- Generative model: GPT-2
- Dataset loaded from JSON file

## How it Works
1. Documents are embedded using BGE-micro
2. Embeddings are stored in a custom vector database
3. A user query is embedded and matched using similarity search
4. Retrieved documents are passed as context to GPT-2
5. GPT-2 generates a response grounded in retrieved data

## How to Run
```bash
cd rag_project
py -3.11 main.py
