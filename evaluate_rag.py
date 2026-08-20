"""
evaluate_rag.py

Script evaluasi kualitas RAG chatbot memakai RAGAS (https://docs.ragas.io).

Untuk tiap pertanyaan di eval_dataset.json, script ini:
  1. Menjalankannya lewat RAG chain yang SAMA dengan app.py (rag_chain.init_rag_chain)
  2. Mengambil jawaban LLM + potongan konteks yang diambil retriever
  3. Menilainya dengan metrik RAGAS:
       - faithfulness      : apakah jawaban benar-benar didukung konteks yang diambil
                              (skor rendah = indikasi halusinasi)
       - answer_relevancy  : apakah jawaban benar-benar menjawab pertanyaan yang diajukan
       - context_precision : apakah konteks yang diambil retriever relevan/berguna
       - context_recall    : apakah retriever berhasil mengambil semua info yang dibutuhkan
                              (HANYA dihitung untuk pertanyaan yang diisi "ground_truth"-nya)

Cara pakai:
    1. pip install ragas pandas
    2. Isi/ubah pertanyaan di eval_dataset.json (ground_truth boleh dikosongkan)
    3. Pastikan Ollama & Pinecone sudah terkonfigurasi sama seperti untuk app.py (.env)
    4. python evaluate_rag.py
    5. Lihat ragas_report.csv untuk skor per pertanyaan

Catatan: RAGAS memakai LLM sebagai "hakim" untuk menghitung sebagian besar metrik ini.
Model Ollama kecil kadang kurang taat pada format instruksi yang dipakai RAGAS secara
internal, jadi skor bisa lebih "berisik" dibanding memakai model besar/hosted. Jika hasil
terasa tidak konsisten, coba set env var RAGAS_EVAL_MODEL ke model Ollama yang lebih besar
daripada model chat sehari-hari (lihat get_evaluator_llm di bawah).
"""

import os
import sys
import json

import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from config import get_embedding_model
from rag_chain import init_rag_chain

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall

# Nama metrik answer-relevancy sempat berganti nama di versi ragas yang lebih baru
# (AnswerRelevancy -> ResponseRelevancy). Coba nama baru dulu, fallback ke nama lama
# supaya script ini tetap jalan di beberapa versi ragas.
try:
    from ragas.metrics import ResponseRelevancy as AnswerRelevancyMetric
except ImportError:
    from ragas.metrics import AnswerRelevancy as AnswerRelevancyMetric

load_dotenv()

EVAL_DATASET_PATH = "eval_dataset.json"
REPORT_PATH = "ragas_report.csv"

# Ollama lokal cenderung lebih lambat & lebih gampang timeout dibanding API hosted,
# jadi worker paralel dikurangi & timeout diperbesar dibanding default RAGAS
# (default: max_workers=16, timeout=180).
RAGAS_RUN_CONFIG = RunConfig(max_workers=2, timeout=300)


def load_eval_questions(path: str = EVAL_DATASET_PATH):
    if not os.path.exists(path):
        print(f"❌ File '{path}' tidak ditemukan. Buat dulu file ini (lihat contoh eval_dataset.json).")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        try:
            questions = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Format JSON di '{path}' tidak valid: {e}")
            sys.exit(1)

    if not questions:
        print(f"❌ '{path}' masih kosong. Tambahkan minimal satu pertanyaan.")
        sys.exit(1)

    return questions


def get_evaluator_llm():
    """LLM 'hakim' RAGAS. Default memakai model Ollama yang sama dengan chat (OLLAMA_MODEL),
    tapi bisa dioverride lewat env var RAGAS_EVAL_MODEL supaya evaluasi bisa memakai model
    yang lebih besar/taat instruksi daripada model untuk chat sehari-hari."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    eval_model = os.getenv("RAGAS_EVAL_MODEL", os.getenv("OLLAMA_MODEL", "gemma4:e4b"))
    llm = ChatOllama(base_url=base_url, model=eval_model, temperature=0.0)
    return LangchainLLMWrapper(llm)


def get_evaluator_embeddings():
    # Pakai ulang embedding model Pinecone yang sama dengan retriever di rag_chain.py
    return LangchainEmbeddingsWrapper(get_embedding_model())


def build_samples(rag_chain, questions):
    samples = []
    for i, item in enumerate(questions, start=1):
        question = item["question"]
        ground_truth = (item.get("ground_truth") or "").strip()

        print(f"[{i}/{len(questions)}] Menjalankan RAG chain untuk: {question}")
        try:
            result = rag_chain.invoke({"input": question})
        except Exception as e:
            print(f"  ⚠️  Dilewati, RAG chain gagal untuk pertanyaan ini: {e}")
            continue

        answer = result.get("answer", "")
        retrieved_contexts = [doc.page_content for doc in result.get("context", [])]

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_contexts,
                reference=ground_truth if ground_truth else None,
            )
        )
    return samples


def build_metrics(evaluator_llm, evaluator_embeddings, has_ground_truth: bool):
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancyMetric(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]
    if has_ground_truth:
        metrics.append(LLMContextRecall(llm=evaluator_llm))
    else:
        print("ℹ️  Tidak ada 'ground_truth' di eval_dataset.json, context_recall dilewati.")
        print("   Isi field ground_truth di eval_dataset.json untuk mengaktifkannya.")
    return metrics


def main():
    questions = load_eval_questions()
    has_ground_truth = any((q.get("ground_truth") or "").strip() for q in questions)

    print("🔧 Menyiapkan RAG chain (Pinecone + Ollama)...")
    rag_chain = init_rag_chain()

    print("🧪 Menjalankan tiap pertanyaan lewat RAG chain...")
    samples = build_samples(rag_chain, questions)
    if not samples:
        print("❌ Tidak ada pertanyaan yang berhasil dijalankan, evaluasi dibatalkan.")
        sys.exit(1)
    dataset = EvaluationDataset(samples=samples)

    print("⚖️  Menyiapkan evaluator LLM & embeddings...")
    evaluator_llm = get_evaluator_llm()
    evaluator_embeddings = get_evaluator_embeddings()
    metrics = build_metrics(evaluator_llm, evaluator_embeddings, has_ground_truth)

    print("📊 Menjalankan evaluasi RAGAS (bisa makan waktu, terutama dengan Ollama lokal)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=RAGAS_RUN_CONFIG,
            raise_exceptions=False,
        )
    except Exception as e:
        print(f"❌ Evaluasi RAGAS gagal: {e}")
        print("💡 Coba: pastikan server Ollama menyala, kurangi jumlah pertanyaan,")
        print("   naikkan timeout di RAGAS_RUN_CONFIG, atau set RAGAS_EVAL_MODEL ke model lain.")
        sys.exit(1)

    df = result.to_pandas()
    df.to_csv(REPORT_PATH, index=False)

    print("\n=== Rata-rata skor ===")
    print(df.mean(numeric_only=True))

    if "faithfulness" in df.columns:
        worst = df.sort_values("faithfulness", ascending=True).head(3)
        print("\n=== 3 jawaban dengan skor faithfulness terendah (paling berpotensi halusinasi) ===")
        for _, row in worst.iterrows():
            print(f"  ({row['faithfulness']:.2f}) {row['user_input']}")

    print(f"\n✅ Detail lengkap per pertanyaan disimpan di '{REPORT_PATH}'")


if __name__ == "__main__":
    main()
