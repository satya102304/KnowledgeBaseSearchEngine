# Knowledge-base Search Engine

## Objective
A simple RAG system that retrieves relevant document content and synthesizes answers using LLMs.

## Features
- Document ingestion (PDF/TXT)
- Embedding-based retrieval (FAISS + SentenceTransformer)
- Answer synthesis via OpenAI GPT model
- Streamlit web interface

## Usage
1. Place your PDFs/texts in `docs/`
2. Run: `streamlit run app.py`
3. Ask your question and get synthesized answers

## Requirements
- Python 3.8+
- OpenAI API key (export as env var: `export OPENAI_API_KEY="your_key"`)
