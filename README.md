## Manual Retrieval-Augmented Generation (RAG)

This project implements a **RAG pipeline from scratch** without using FAISS or LangChain.  
The goal is to compare a raw LLM against a retrieval-augmented LLM and analyze how retrieval improves factual grounding.

---

## What This Implements

- Manual document chunking  
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`)  
- Manual cosine similarity  
- Top-k retrieval  
- Prompt construction with retrieved context  
- Generation using `TinyLlama-1.1B-Chat`  
- Comparison: **Raw LLM vs RAG**

The knowledge base contains information about Queen’s University and is used to answer factual queries.

---

## Example Comparison

For the query:

> *How many students are at Queen’s University?*

- **RAG** retrieves a relevant chunk and correctly outputs:  
  **33,842 students**
- **Raw LLM** produces arbitrary numbers (hallucination)

This demonstrates how retrieval grounds responses in source text.

---

## How to Run

###  Install Dependencies
```bash
pip install sentence-transformers
pip install transformers accelerate

Then run the script which will:
-Chunk the knowledge base
-Generate embeddings
-Retrieve top-k relevant chunks
-Generate both: RAG answer, Raw LLM answer
-Print a side-by-side comparison
