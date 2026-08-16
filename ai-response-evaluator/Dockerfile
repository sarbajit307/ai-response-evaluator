FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (build-essential for FAISS etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

# Copy a startup script to run both
RUN chmod +x docker-entrypoint.sh || true

CMD ["python", "-m", "backend.app.main"]
