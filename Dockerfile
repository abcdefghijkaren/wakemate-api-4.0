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

COPY . .

ENV DATABASE_URL=postgresql://wakemate_user:neNuQ9GMsK7lahBvAsM5b9atg9ijsrwI@dpg-d1pm85ruibrs73dt6bpg-a.oregon-postgres.render.com/user_info_wakemate_db
ENV PORT=10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]