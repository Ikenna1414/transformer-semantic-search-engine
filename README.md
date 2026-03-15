# Semantic Search for Research Papers using Transformers and FAISS

This project implements a semantic search engine for research papers using transformer-based embeddings and vector similarity search.

Instead of traditional keyword matching, the system retrieves papers based on semantic meaning. A natural language query is converted into an embedding using a Sentence Transformer model, and FAISS is used to efficiently retrieve the most relevant research papers.

The system demonstrates how modern NLP embeddings can be used to build scalable information retrieval systems.

## Project Overview

The pipeline consists of four main steps:

1. Data Collection  
Research papers are collected from the arXiv API, including title, abstract, authors, and subject tags.

2. Text Embedding  
Paper abstracts are converted into dense vector embeddings using the Sentence Transformers model `all-MiniLM-L6-v2`.

3. Vector Indexing  
Embeddings are stored in a FAISS vector index to allow fast similarity search across thousands of documents.

4. Semantic Query Search  
A user query is encoded into an embedding and the FAISS index returns the most semantically similar papers.

## Technologies Used

Python  
Sentence Transformers  
FAISS (Facebook AI Similarity Search)  
NumPy  
arXiv API

## Installation

Install required libraries:

pip install sentence-transformers faiss-cpu arxiv numpy

## Running the Project

1. Fetch research papers from arXiv

Example:

search = arxiv.Search(
    query="machine learning",
    max_results=1000
)

2. Generate embeddings using Sentence Transformers

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

3. Build FAISS vector index

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

4. Perform semantic search

results = search("stochastic optimization methods for deep learning")

The system returns the most semantically relevant papers based on vector similarity.

## Example Query

Query:
stochastic optimization methods for deep learning

Example retrieved results include research papers related to stochastic gradient methods, optimization techniques, and deep learning training dynamics.

## Key Idea

Traditional search relies on keyword matching.  
Semantic search represents text as dense vector embeddings in high-dimensional space, allowing the system to retrieve relevant documents even when different wording or phrasing is used.

## Future Improvements

Scale the system to millions of documents  
Add GPU acceleration  
Implement approximate nearest neighbor indexing  
Build a web interface for interactive semantic search

