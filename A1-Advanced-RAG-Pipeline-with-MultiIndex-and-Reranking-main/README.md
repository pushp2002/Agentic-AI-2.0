### Advanced-RAG-Pipeline-with-MultiIndex-and-Reranking

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using multiple PDF files containing text, images, and tables. The pipeline includes semantic chunking, embedding using HuggingFace models, and storage in vector databases. It supports three FAISS-based indexing strategies (Flat, HNSW, and IVF) to evaluate retrieval speed and accuracy. Additionally, it integrates reranking using BM25, prompt engineering for LLM-based generation, and exports results into DOCX reports.

## Features

- Parses large PDF documents (over 200 pages)
- Applies semantic chunking for context-aware embedding
- Uses sentence-transformer embeddings
- Stores vectorized data using FAISS with three indexing strategies:
  - Flat Index
  - HNSW (Hierarchical Navigable Small World)
  - IVF (Inverted File Index)
- Benchmarks retrieval time for each indexing method
- Evaluates accuracy of similarity search results
- Implements reranking using BM25
- Uses prompt templates to generate responses via LLM
- Renders generated outputs to DOCX files

## Input Files

- Global report on diabetes.pdf
- Understanding_Science.pdf

These PDFs are extracted, chunked, embedded, and indexed before retrieval and response generation.

## Output Files

- medical_report.docx
- science_report.docx

These Word documents contain the final output generated from the retrieved context using a language model.

## Steps to Run

1. Clone the Repository
2. Install Dependencies
3. Set Environment Variables
4. Add your API keys in .env file
5. Run the Notebooks
- Complete RAG NOTEBOOK `rag_NOTEBOOK.ipynb` .
- Execute `assignment.ipynb` for initial RAG setup and full implementation.

## Retrieval Benchmarking

All three FAISS index types were tested and compared based on retrieval speed and output accuracy. Reranking with BM25 was applied to enhance final results before LLM generation.

## Deliverables

- Executable notebooks
- DOCX reports with LLM-generated content



## By Shohrab Haque Shahil
