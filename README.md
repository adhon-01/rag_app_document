# Production-Grade Local RAG System

A Retrieval-Augmented Generation (RAG) system for querying private documents, combining hybrid retrieval (dense + BM25), Cohere re-ranking, and automated evaluation with RAGAS. Built with Ollama, Pinecone, Cohere, and LangChain.

---

## 🌟 Overview & Key Benefits

Standard RAG implementations often suffer from accuracy drops, missing keyword context, and hallucinated answers. This project addresses those limitations with:

* **Hybrid Retrieval (Dense + Sparse):** Combines Pinecone dense vector search with a local BM25 retriever (`rank_bm25`), merged via LangChain's `EnsembleRetriever` — captures both deep semantic context and exact keyword matches.
* **Cross-Encoder Re-ranking:** Chunks retrieved by the ensemble are re-scored with Cohere Rerank (`rerank-multilingual-v3.0`) before being passed to the LLM.
* **Grounded Answers:** The system prompt (managed in `prompts.yaml`, loaded at runtime — not hardcoded) instructs the LLM to answer strictly from retrieved context and to say so explicitly when the answer isn't in the documents.
* **Source Attribution in UI:** Each answer in the Streamlit app has an expandable "📎 Sumber jawaban ini" section listing the source file (and page number, for PDFs) of the chunks actually used to generate it.
* **Prompt Engineering Management:** System prompts are decoupled from application logic and versioned inside `prompts.yaml`.
* **Automated Web-based Ingestion:** Upload files (`.pdf`, `.txt`, `.docx`) directly through the Streamlit interface with instant background embedding and Pinecone indexing.
* **Automated Evaluation:** `evaluate_rag.py` scores the pipeline against `eval_dataset.json` using RAGAS (faithfulness, answer relevancy, context precision/recall) and writes results to `ragas_report.csv`.

> **Note on hybrid retrieval design:** this project intentionally does *not* use Pinecone's native single-vector hybrid search (sparse+dense combined in one index, which requires a `dotproduct`-metric index). The Pinecone index here is created with `metric="cosine"` (see `ingest.py`) and already holds ingested data — changing the metric would mean deleting the index and re-ingesting everything. Instead, hybrid retrieval is implemented at the LangChain layer via `EnsembleRetriever`, which is functionally equivalent without touching the existing index.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend / UI:** Streamlit
* **Orchestration:** LangChain
* **LLM Engine:** Ollama (local)
* **Dense retrieval:** Pinecone (Serverless, `cosine` metric)
* **Embedding Model:** Pinecone Embeddings (`multilingual-e5-large`)
* **Sparse retrieval:** `rank_bm25` via LangChain's `BM25Retriever`, fit on the local `data_dokumen/` corpus
* **Hybrid merge:** LangChain `EnsembleRetriever`
* **Re-ranker:** Cohere (`rerank-multilingual-v3.0`) via `ContextualCompressionRetriever`
* **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision, context recall)
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
   (double-check this exact model tag exists in your local Ollama — adjust `OLLAMA_MODEL` in `.env` if it doesn't.)
3. API Keys for **Pinecone** and **Cohere** (free tiers available).

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
   ```
   All dependencies — including `rank_bm25`, `langchain-cohere`, and `pyyaml` — are already listed in `requirements.txt`; no extra manual install step is needed.

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

1. **(Optional) Bulk-ingest local documents**
   Put files into `data_dokumen/`, then:
   ```bash
   python ingest.py
   ```

2. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

3. **Ingest documents via the UI**
   * Open the sidebar in your browser (`http://localhost:8501`).
   * Upload `.pdf`, `.txt`, or `.docx` files.
   * Click **"Proses & Ingest Dokumen"**.

4. **Ask questions**
   * Type your query in the chat input.
   * Expand **"📎 Sumber jawaban ini"** below each answer to see which source file(s)/page(s) it's grounded in.

5. **Run evaluation**
   ```bash
   python evaluate_rag.py
   ```
   See `ragas_report.csv` for per-question scores.

---

## 📁 Repository Structure

```text
├── app.py              # Streamlit UI, chat loop, source/citation display
├── ingest.py            # Document loading, chunking, Pinecone ingestion (helpers reused by rag_chain.py & app.py)
├── rag_chain.py          # Hybrid retriever (dense + BM25) + Cohere reranker + LLM chain
├── evaluate_rag.py       # RAGAS evaluation script
├── config.py             # Embedding model configuration
├── prompts.yaml          # System prompt (loaded at runtime by rag_chain.py)
├── eval_dataset.json     # Evaluation questions for RAGAS
├── .env                  # API keys and environment variables (ignored by Git)
└── .gitignore            # Rules for ignoring local indexes and sensitive data
```

---

## ⚠️ Known limitations

* `eval_dataset.json` currently only covers one of the documents in `data_dokumen/` (a journal article on an attendance system) — evaluation coverage should be expanded to the other documents there for a more representative score.
* `bm25_values.json` in the repo root is a leftover artifact from an earlier native-Pinecone-hybrid attempt (a fitted `pinecone-text` BM25 encoder, tokenized with English-language settings). It is **not used** by the current pipeline (which fits a fresh, language-agnostic BM25 index from `data_dokumen/` at runtime instead) — safe to delete.
