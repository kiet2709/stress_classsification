# Dockerfile
FROM python:3.9-slim

# Cài gcc (cần cho transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# CÀI TORCH CPU (phiên bản chính thức, không cần +cpu)
# 2.7.0 trở lên đã có sẵn CPU build trên PyPI
RUN pip install --no-cache-dir torch==2.7.0

# Copy code
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
