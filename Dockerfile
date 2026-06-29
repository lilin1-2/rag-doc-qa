FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖（chromadb 需要 sqlite3 >= 3.35）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 首次启动时下载 Embedding 模型比较慢（约 400MB），
# 用 docker-compose 挂载卷可以避免每次重启都重新下载
COPY . .

EXPOSE 8001
EXPOSE 8501

# 默认启动 FastAPI，想用 Streamlit 界面时改成下面那行
CMD ["python", "app.py"]
# CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"]
