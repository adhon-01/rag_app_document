"""
rag_chain.py

Berisi logic untuk membangun RAG chain (retriever Pinecone + LLM Ollama + prompt).
Dipisah dari app.py supaya bisa dipakai bersama oleh:
  - app.py            (aplikasi chat Streamlit)
  - evaluate_rag.py    (script evaluasi RAGAS)
tanpa perlu ikut menjalankan kode UI Streamlit (st.set_page_config, st.sidebar, dst)
setiap kali modul ini di-import.
"""

import os
from dotenv import load_dotenv
from config import get_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

SYSTEM_PROMPT = (
    "Anda adalah asisten AI yang cerdas dan jujur.\n"
    "Jawablah pertanyaan pengguna hanya berdasarkan potongan konteks yang disediakan di bawah ini.\n"
    "Jika informasi di konteks tidak cukup untuk menjawab, katakan bahwa Anda tidak tahu jawabannya secara sopan.\n\n"
    "Konteks:\n{context}"
)


def get_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    return ChatOllama(base_url=base_url, model=model_name, temperature=0.2)


def get_retriever(k: int = 4):
    embeddings = get_embedding_model()
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def init_rag_chain():
    """Membangun retrieval chain lengkap. Output invoke()/stream() berisi
    minimal key 'context' (list Document hasil retrieval) dan 'answer' (jawaban LLM) —
    dua-duanya dipakai lagi oleh evaluate_rag.py untuk evaluasi RAGAS."""
    retriever = get_retriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
