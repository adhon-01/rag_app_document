"""
rag_chain.py

Berisi logic untuk membangun RAG chain:
  - Dense retriever (Pinecone) + Sparse retriever (BM25 lokal) digabung lewat EnsembleRetriever
  - Hasil ensemble di-rerank pakai Cohere Rerank (ContextualCompressionRetriever)
  - LLM: Ollama
  - System prompt di-load dari prompts.yaml (bukan hardcoded), supaya README & kode
    selalu sinkron

Dipisah dari app.py supaya bisa dipakai bersama oleh:
  - app.py            (aplikasi chat Streamlit)
  - evaluate_rag.py    (script evaluasi RAGAS)
tanpa perlu ikut menjalankan kode UI Streamlit (st.set_page_config, st.sidebar, dst)
setiap kali modul ini di-import.

Catatan desain hybrid retrieval:
Sengaja TIDAK memakai native hybrid search Pinecone (sparse+dense dalam satu vector,
butuh index dengan metric="dotproduct"), karena index yang sudah ada di project ini
dibuat dengan metric="cosine" (lihat ingest.py) dan sudah berisi data ter-ingest.
Mengubah metric index berarti index harus dihapus & semua data di-ingest ulang -
perubahan destruktif yang sebaiknya dilakukan sadar/manual oleh pemilik project,
bukan otomatis lewat refactor ini. Sebagai gantinya, hybrid retrieval diimplementasi
di level LangChain: BM25Retriever (sparse, lokal, di-fit dari data_dokumen/) + retriever
Pinecone (dense) digabung lewat EnsembleRetriever - hasilnya setara secara fungsi,
tanpa menyentuh index Pinecone yang sudah ada.
"""

import os
import yaml
from dotenv import load_dotenv
from config import get_embedding_model
from ingest import load_and_chunk_folder, DATA_DIR
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts.yaml")
_FALLBACK_SYSTEM_PROMPT = (
    "Anda adalah asisten AI yang cerdas dan jujur.\n"
    "Jawablah pertanyaan pengguna hanya berdasarkan potongan konteks yang disediakan di bawah ini.\n"
    "Jika informasi di konteks tidak cukup untuk menjawab, katakan bahwa Anda tidak tahu jawabannya secara sopan.\n\n"
    "Konteks:\n{context}"
)


def load_system_prompt() -> str:
    """Baca system prompt dari prompts.yaml supaya prompt benar-benar dikelola
    di satu tempat (bukan hardcoded di sini) - selaras dengan klaim README.
    Fallback ke prompt default kalau file tidak ada/rusak, supaya app tidak crash."""
    try:
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        prompt = data.get("system_prompt") if data else None
        if prompt:
            return prompt
    except (FileNotFoundError, yaml.YAMLError):
        pass
    return _FALLBACK_SYSTEM_PROMPT


def get_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    return ChatOllama(base_url=base_url, model=model_name, temperature=0.2)


def get_dense_retriever(k: int = 4):
    embeddings = get_embedding_model()
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_bm25_retriever(k: int = 4):
    """BM25 retriever lokal, di-fit dari korpus yang sama (data_dokumen/) dengan
    chunking yang sama seperti saat ingest ke Pinecone, supaya hasil ensemble
    setara granularitasnya dengan hasil dense retriever."""
    chunks = load_and_chunk_folder(DATA_DIR)
    if not chunks:
        return None
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def get_retriever(k: int = 4, use_reranker: bool = True):
    dense_retriever = get_dense_retriever(k=k)
    bm25_retriever = get_bm25_retriever(k=k)

    if bm25_retriever is None:
        # data_dokumen/ kosong (misal baru clone project, belum ada dokumen lokal) -
        # fallback ke dense-only supaya app tetap jalan, daripada crash
        base_retriever = dense_retriever
    else:
        base_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

    if not use_reranker or not os.getenv("COHERE_API_KEY"):
        return base_retriever

    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=k)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def init_rag_chain():
    """Membangun retrieval chain lengkap. Output invoke()/stream() berisi
    minimal key 'context' (list Document hasil retrieval) dan 'answer' (jawaban LLM) —
    dua-duanya dipakai lagi oleh evaluate_rag.py untuk evaluasi RAGAS dan oleh app.py
    untuk menampilkan sumber/citation."""
    retriever = get_retriever()
    llm = get_llm()
    system_prompt = load_system_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
