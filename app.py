import os
import streamlit as st
from config import get_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def get_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    return ChatOllama(base_url=base_url, model=model_name, temperature=0.2)

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

# 1. Definisikan fungsi untuk membuat chain
def init_rag_chain():
    embeddings = get_embedding_model()
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), 
        embedding=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    llm = get_llm()
    
    system_prompt = (
        "Anda adalah asisten AI yang cerdas dan jujur.\n"
        "Jawablah pertanyaan pengguna hanya berdasarkan potongan konteks yang disediakan di bawah ini.\n"
        "Jika informasi di konteks tidak cukup untuk menjawab, katakan bahwa Anda tidak tahu jawabannya secara sopan.\n\n"
        "Konteks:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

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