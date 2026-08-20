import os
import streamlit as st
from config import get_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from rag_chain import init_rag_chain

# Load env variables
load_dotenv()

st.set_page_config(page_title="Retrieval Augmented Generation", page_icon="🦙", layout="centered")
st.title("🦙 Ollama Local RAG Chatbot")
st.write("Aplikasi RAG menggunakan Ollama (Gemma) dan Pinecone.")

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
            os.makedirs("data_dokumen", exist_ok=True)
            saved_docs = []
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data_dokumen", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load dokumen berdasarkan Ekstensi
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    saved_docs.extend(loader.load())
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(file_path)
                    saved_docs.extend(loader.load())
                elif uploaded_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    saved_docs.extend(loader.load())
            
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
                # Clear cache agar rantai RAG memuat data terbaru jika diperlukan
                st.cache_resource.clear()
            else:
                st.sidebar.warning("Tidak ada dokumen valid yang terbaca.")

# 1. Chain RAG (retriever + LLM + prompt) sekarang didefinisikan di rag_chain.py
#    supaya bisa dipakai bersama oleh app.py ini dan evaluate_rag.py (evaluasi RAGAS)
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

# Kolom Input Chat User
if user_query := st.chat_input("Tanyakan isi dokumen lokal Anda di sini..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in rag_chain.stream({"input": user_query}):
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memanggil model Ollama: {e}")