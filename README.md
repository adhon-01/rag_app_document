# Production-Grade Local RAG System

A production-grade Retrieval-Augmented Generation (RAG) system designed for querying private documents with high precision, zero-hallucination safeguards, and automated citation tracking. Built with Ollama, Pinecone, Cohere, and LangChain, this application transforms local files into an interactive, verifiable AI knowledge base.

---

## 🌟 Overview & Key Benefits

Standard RAG implementations often suffer from accuracy drops, missing keyword context, and hallucinated answers. This project addresses those limitations by implementing enterprise RAG patterns:

* **Hybrid Search (Dense + Sparse):** Combines semantic vector embeddings with BM25 keyword matching to capture both deep context and exact technical terminology.
* **Cross-Encoder Re-ranking:** Uses Cohere Re-ranker to re-evaluate and re-score retrieved document chunks, passing only the top most relevant snippets to the LLM.
* **Strict Citation Enforcement:** Mandates that the LLM answers strictly from provided document contexts. If information is missing, the system gracefully refuses to answer rather than hallucinating.
* **Direct Paragraph Citations:** Every answer includes exact document references, page numbers, and text snippets for transparent source verification.
* **Prompt Engineering Management:** System prompts are decoupled from application logic and versioned inside `prompts.yaml`.
* **Automated Web-based Ingestion:** Upload files (`.pdf`, `.txt`, `.docx`) directly through the Streamlit interface with instant background embedding and BM25 index updating.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend / UI:** Streamlit
* **Orchestration:** LangChain
* **LLM Engine:** Ollama (Gemma 4:e4b running locally)
* **Vector Database:** Pinecone (Serverless with `dotproduct` metric)
* **Embedding Model:** Pinecone Embeddings (`multilingual-e5-large`)
* **Sparse Encoder:** `pinecone-text` (BM25)
* **Re-ranker:** Cohere (`rerank-multilingual-v3.0`)
* **Prompt Management:** PyYAML
* **Evaluating :** RAGAS

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10+** installed on your machine.
2. **Ollama** installed and running locally with your desired model:
   ```bash
   ollama run gemma4:e4b
   ```
3. API Keys for **Pinecone** and **Cohere** (Free tiers available).

---

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/adhon-01/rag_app_document.git
   cd rag_app_document
   ```

2. **Create a Virtual Environment & Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install pinecone-text nltk langchain-cohere pyyaml
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=rag-hybrid-v1
   COHERE_API_KEY=your_cohere_api_key_here
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=gemma4:e4b
   ```

---

## 💡 Usage

1. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

2. **Ingest Documents**
   * Open the sidebar in your web browser (`http://localhost:8501`).
   * Upload your `.pdf`, `.txt`, or `.docx` files.
   * Click **"Proses & Ingest Dokumen"**. The app will split texts, compute BM25 sparse weights, generate dense vectors, and sync with Pinecone.

3. **Ask Questions**
   * Type your query in the chat input.
   * The assistant will verify retrieved documents, generate an answer, and list exact paragraph citations at the bottom of the response.

---

## 📁 Repository Structure

```text
├── app.py              # Main Streamlit application UI & RAG chain execution
├── ingest.py           # Document loading, chunking, BM25 fitting, & vector ingestion
├── config.py           # Embedding model configurations
├── prompts.yaml        # System prompts and strict anti-hallucination rules
├── .env                # API keys and environment variables (ignored by Git)
└── .gitignore          # Rules for ignoring local indexes and sensitive data
```
