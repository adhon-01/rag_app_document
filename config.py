import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings # Jika menggunakan embedding hosted dari Pinecone

load_dotenv()

def get_embedding_model():
    # Menggunakan model embedding hosted dari Pinecone (contoh: multilingual-e5-large)
    # Pastikan API key Pinecone sudah ada di environment variable
    return PineconeEmbeddings(
        model="multilingual-e5-large", 
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )