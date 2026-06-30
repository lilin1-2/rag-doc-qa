"""
RAG 文档问答服务 —— FastAPI 接口层

接口：
  GET  /health       健康检查
  POST /rag/ask       文档问答（输入文本直接问答）
  POST /rag/upload    上传文档文本，自动建向量库
"""
from fastapi import FastAPI, HTTPException, Request, Depends, Header
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_core import build_collection, get_collection, ask

# API 认证
def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key 无效或缺失")
    return True

app = FastAPI(
    title="RAG 文档问答服务",
    description="基于 Chroma + DeepSeek 的 RAG 问答 API",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

collection = None


@app.on_event("startup")
def startup():
    global collection
    try:
        collection = get_collection("pdf_knowledge")
        logger.info(f"已加载向量库，{collection.count()} 条记录")
    except:
        logger.warning("未找到向量库，请先调用 /rag/upload 上传文档")


# ============================================================
# 数据模型
# ============================================================

class AskRequest(BaseModel):
    text: str = Field(..., description="文档文本或问题", min_length=1, max_length=50000)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list


class UploadRequest(BaseModel):
    text: str = Field(..., description="文档文本内容", min_length=1)


# ============================================================
# 接口
# ============================================================

@app.get("/health")
def health():
    try:
        import requests as _r
        _r.get(DEEPSEEK_URL.replace("/chat/completions", ""), timeout=5)
        api_status = "connected"
    except:
        api_status = "unreachable"
    return {
        "status": "ok",
        "api": api_status,
        "collection_ready": collection is not None,
        "doc_count": collection.count() if collection else 0,
    }


@app.post("/rag/upload", dependencies=[Depends(verify_api_key)])
def upload(req: UploadRequest):
    global collection
    collection, count = build_collection(req.text, "pdf_knowledge")
    return {"message": "文档处理完成", "collection_name": "pdf_knowledge", "chunk_count": count}


@app.post("/rag/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def rag_ask(req: AskRequest):
    if collection is None:
        raise HTTPException(status_code=400, detail="暂无向量库，请先调用 /rag/upload 上传文档")
    try:
        result = ask(collection, req.text)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RAG 问答失败：{e}")


@app.exception_handler(Exception)
def global_exception(request: Request, exc: Exception):
    logger.error(f"未捕获异常: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
