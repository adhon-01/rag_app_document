import os
import streamlit as st
from config import get_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from ingest import load_single_document, DATA_DIR
from rag_chain import init_rag_chain

# Load env variables
load_dotenv()

st.set_page_config(page_title="Retrieval Augmented Generation", page_icon="🦙", layout="centered")
st.title("🦙 Ollama Local RAG Chatbot")
st.write("Aplikasi RAG menggunakan Ollama (Gemma), Pinecone, hybrid retrieval (BM25 + dense), dan Cohere reranker.")

# --- FITUR UPLOAD DOKUMEN VIA STREAMLIT ---
st.sidebar.header("📁 Tambah Dokumen Baru")
uploaded_files = st.sidebar.file_uploader(
    "Pilih file (.pdf, .txt, .docx)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Proses & Ingest Dokumen"):
        with st.spinner("Sedang memproses dan mengunggah ke Pinecone..."):
            os.makedirs(DATA_DIR, exist_ok=True)
            saved_docs = []

            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Loader dipusatkan di ingest.py (load_single_document) supaya tidak
                # duplikat logic if/elif ekstensi file dengan ingest.py
                saved_docs.extend(load_single_document(file_path))

            if saved_docs:
                # Chunking
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
                chunks = text_splitter.split_documents(saved_docs)

                # Embedding & Upload ke Pinecone
                embeddings = get_embedding_model()
                PineconeVectorStore.from_documents(
                    chunks,
                    embeddings,
                    index_name=os.getenv("PINECONE_INDEX_NAME")
                )
                st.sidebar.success(f"Berhasil mengunggah dan meng-ingest {len(chunks)} chunks baru!")
                # Clear cache: dokumen baru berarti retriever (dense & BM25) & chain
                # perlu dibangun ulang supaya ikut mengindeks dokumen barusan
                st.cache_resource.clear()
            else:
                st.sidebar.warning("Tidak ada dokumen valid yang terbaca.")

# 1. Chain RAG (retriever hybrid + reranker + LLM + prompt) didefinisikan di
#    rag_chain.py supaya bisa dipakai bersama oleh app.py ini dan evaluate_rag.py
@st.cache_resource
def get_cached_rag_chain():
    return init_rag_chain()

try:
    rag_chain = get_cached_rag_chain()
except Exception as e:
    st.error(f"Gagal inisialisasi sistem RAG: {e}")
    st.stop()

# Riwayat Chat di UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sumber jawaban ini"):
                for s in message["sources"]:
                    st.markdown(f"- {s}")

# Kolom Input Chat User
if user_query := st.chat_input("Tanyakan isi dokumen lokal Anda di sini..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        retrieved_docs = []

        try:
            for chunk in rag_chain.stream({"input": user_query}):
                if "context" in chunk:
                    retrieved_docs = chunk["context"]
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Tampilkan sumber (nama file + nomor halaman jika ada) dari dokumen
            # yang benar-benar dipakai untuk menjawab pertanyaan ini
            sources = []
            for doc in retrieved_docs:
                src = os.path.basename(doc.metadata.get("source", "Tidak diketahui"))
                page = doc.metadata.get("page")
                label = f"{src} (hal. {page + 1})" if page is not None else src
                if label not in sources:
                    sources.append(label)

            if sources:
                with st.expander("📎 Sumber jawaban ini"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memanggil model Ollama: {e}")
