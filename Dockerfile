# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libpq-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 複製整個專案（包含 app/ 資料夾）
COPY . .

# 設定環境變數（Render 上會 override）
ENV DATABASE_URL=postgresql://postgres:123456@localhost:5432/user_info_wakemate_db
ENV PORT=10000

# 指定執行 app.main:app（從 app/main.py 取出 app）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]