import os
import time
from config import get_embedding_model
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def run_ingestion():
    print("🔄 Memulai proses ingest data dari folder lokal...")
    
    # 1. Koneksi dan Validasi Indeks Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    pc = Pinecone(api_key=api_key)
    
    # Ambil daftar indeks yang ada di dasbor Pinecone Anda
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    # Jika indeks belum ada di dasbor, buat otomatis secara otomatis
    if index_name not in existing_indexes:
        print(f"📁 Indeks '{index_name}' tidak ditemukan di dasbor Pinecone.")
        print("⚡ Membuat indeks baru secara otomatis (768 Dimensi, Serverless)...")
        
        pc.create_index(
            name=index_name,
            dimension=1024, # Wajib 1024 untuk model embedding Pinecone multilingual-e5-large
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Region gratis default Pinecone Serverless
            )
        )
        # Tunggu beberapa detik sampai server Pinecone siap menerima data
        print("⏳ Menunggu indeks siap di server Pinecone...")
        time.sleep(10)
    else:
        print(f"✨ Indeks '{index_name}' ditemukan dan siap digunakan.")
    
    # 2. Ambil dokumen dari folder lokal 'data_dokumen'
    print("📥 Membaca file dari folder data_dokumen/...")
    if not os.path.exists("data_dokumen"):
        os.makedirs("data_dokumen")
        print("📁 Folder 'data_dokumen' baru saja dibuat. Taruh file Anda di sana!")
        return

    pdf_loader = DirectoryLoader("data_dokumen/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader("data_dokumen/", glob="**/*.txt", loader_cls=TextLoader)
    word_loader = DirectoryLoader("data_dokumen/", glob="**/*.docx", loader_cls=Docx2txtLoader)
    
    docs = pdf_loader.load() + txt_loader.load() + word_loader.load()
    print(f"📄 Berhasil memuat {len(docs)} halaman/dokumen dari lokal.")
    
    if not docs:
        print("❌ Tidak ada file dokumen (.pdf / .txt / .docx) di dalam folder 'data_dokumen/'.")
        return

    # 3. Proses Chunking (Pemotongan Teks)
    print("✂️ Memotong dokumen menjadi beberapa bagian (chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=60
    )
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Dokumen berhasil dipecah menjadi {len(chunks)} chunks.")

    # 4. Ambil embedding model & Upload ke Pinecone
    print("🚀 Mengonversi teks ke vektor dan mengunggah ke Pinecone...")
    embeddings = get_embedding_model()
    
    # Simpan potongan teks ke Pinecone
    PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=index_name
    )
    print("✅ Ingest data selesai! Vektor siap digunakan di Pinecone.")

if __name__ == "__main__":
    run_ingestion()