"""
RAG 核心引擎 —— 文档加载、向量化、检索、生成

独立模块，不依赖 FastAPI，可单独测试。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import API_KEY, DEEPSEEK_URL

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ============================================================
# 文档处理
# ============================================================

def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def split_chunks(text: str, chunk_size=500, chunk_overlap=50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


# ============================================================
# 向量库管理
# ============================================================

def build_collection(text: str, collection_name: str = "pdf_knowledge"):
    chunks = split_chunks(text)
    embeddings = model.encode(chunks)

    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    col = chroma_client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    col.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return col, len(chunks)


def get_collection(name: str = "pdf_knowledge"):
    return chroma_client.get_collection(name=name)


# ============================================================
# 检索 + 生成
# ============================================================

def retrieve(collection, query: str, top_k: int = 3):
    qv = model.encode([query])
    results = collection.query(query_embeddings=qv.tolist(), n_results=top_k)
    chunks = list(results["documents"][0])
    sources = [
        {"chunk_id": results["ids"][0][i],
         "distance": round(results["distances"][0][i], 4),
         "text_preview": c[:100]}
        for i, c in enumerate(chunks)
    ]
    return chunks, sources


def generate(question: str, context_chunks: list) -> str:
    ctx = "".join(f"\n[参考资料 {i+1}]\n{c}" for i, c in enumerate(context_chunks))
    system = f"严格根据以下参考资料回答。没提到就说'文档中没有提到'，不要编造。\n参考资料：\n{ctx}"

    resp = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        },
        timeout=30,
        proxies={"http": None, "https": None},  # 跳过公司代理
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ============================================================
# 一键问答
# ============================================================

def ask(collection, question: str, top_k: int = 3) -> dict:
    chunks, sources = retrieve(collection, question, top_k)
    answer = generate(question, chunks)
    return {"question": question, "answer": answer, "sources": sources}
