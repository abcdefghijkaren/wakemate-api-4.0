# 使用 Python 3.13 作為基礎鏡像
FROM python:3.13-slim

# 設定工作目錄 (/app)
WORKDIR /app

# 1️⃣ 先複製 requirements.txt (Docker 會緩存此層，避免每次修改代碼都要重新安裝依賴)
COPY requirements.txt .

# 2️⃣ 安裝依賴項 (需要 PostgreSQL 的 psycopg2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libpq-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 3️⃣ 複製 **整個專案** 到容器
COPY PYTHON_FASTAPI .

# 4️⃣ 設定環境變數 (預設值，可以在部署時 override)
ENV DATABASE_URL=postgresql://user:password@localhost:5432/fastapi
ENV PORT=10000  
# Render 預設使用的 PORT

# 5️⃣ 啟動 FastAPI (uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]