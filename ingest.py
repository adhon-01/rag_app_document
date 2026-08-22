import os
import time
from config import get_embedding_model
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

DATA_DIR = "data_dokumen"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60


def load_single_document(file_path: str):
    """Load satu file dokumen (.pdf/.txt/.docx) sesuai ekstensinya.
    Dipakai bareng oleh run_ingestion() (bulk folder) dan app.py (upload per-file)
    supaya logic loading tidak diduplikasi di dua tempat."""
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    return []


def load_documents_from_folder(folder: str = DATA_DIR):
    """Load semua dokumen (.pdf/.txt/.docx) di dalam folder lokal."""
    if not os.path.exists(folder):
        return []
    pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader)
    word_loader = DirectoryLoader(folder, glob="**/*.docx", loader_cls=Docx2txtLoader)
    return pdf_loader.load() + txt_loader.load() + word_loader.load()


def load_and_chunk_folder(folder: str = DATA_DIR):
    """Load + chunk semua dokumen di folder lokal. Dipakai oleh run_ingestion()
    (upload ke Pinecone) dan rag_chain.py (fit BM25Retriever) - chunk_size/overlap
    disamakan supaya granularitas hasil dense & sparse retriever setara."""
    docs = load_documents_from_folder(folder)
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def run_ingestion():
    print("🔄 Memulai proses ingest data dari folder lokal...")

    # 1. Koneksi dan Validasi Indeks Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"📁 Indeks '{index_name}' tidak ditemukan di dasbor Pinecone.")
        print("⚡ Membuat indeks baru secara otomatis (1024 Dimensi, Serverless)...")

        pc.create_index(
            name=index_name,
            dimension=1024,  # Wajib 1024 untuk model embedding Pinecone multilingual-e5-large
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Region gratis default Pinecone Serverless
            )
        )
        print("⏳ Menunggu indeks siap di server Pinecone...")
        time.sleep(10)
    else:
        print(f"✨ Indeks '{index_name}' ditemukan dan siap digunakan.")

    # 2. Ambil & chunk dokumen dari folder lokal
    print(f"📥 Membaca file dari folder {DATA_DIR}/...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Folder '{DATA_DIR}' baru saja dibuat. Taruh file Anda di sana!")
        return

    docs = load_documents_from_folder(DATA_DIR)
    print(f"📄 Berhasil memuat {len(docs)} halaman/dokumen dari lokal.")

    if not docs:
        print(f"❌ Tidak ada file dokumen (.pdf / .txt / .docx) di dalam folder '{DATA_DIR}/'.")
        return

    print("✂️ Memotong dokumen menjadi beberapa bagian (chunking)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"🧩 Dokumen berhasil dipecah menjadi {len(chunks)} chunks.")

    # 3. Ambil embedding model & Upload ke Pinecone
    print("🚀 Mengonversi teks ke vektor dan mengunggah ke Pinecone...")
    embeddings = get_embedding_model()

    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name
    )
    print("✅ Ingest data selesai! Vektor siap digunakan di Pinecone.")


if __name__ == "__main__":
    run_ingestion()
