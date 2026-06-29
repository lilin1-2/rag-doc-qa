# RAG 文档问答机器人

## 简介

基于 Chroma + DeepSeek 的 RAG 文档问答系统。上传文档后自动切分向量化，用自然语言提问即可获取基于文档的精准回答。

## 核心流程

```
文档文本 → RecursiveCharacterTextSplitter 切分 → Embedding 向量化 → Chroma 存储
                                                                    ↓
用户问题 → 向量化 → 检索 Top-3 chunks → 拼接 prompt → DeepSeek 生成答案
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/rag/upload` | 上传文档文本，自动处理并建库 |
| POST | `/rag/ask` | RAG 问答 |

## 快速开始

```bash
pip install -r requirements.txt
cp config_example.py config.py
# 编辑 config.py 填入 API Key
python app.py
```

## 测试

```bash
# 上传公司文档
curl -X POST http://localhost:8001/rag/upload \
  -H "Content-Type: application/json" \
  -d '{"text":"公司简介...2026年营收目标5000万..."}'

# 问答
curl -X POST http://localhost:8001/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"text":"公司的营收目标是多少？"}'
```

## 技术栈

Python · FastAPI · Chroma · Sentence-Transformers · DeepSeek API · LangChain Text Splitters · PyPDF
