# Docker Compose file
version: '3.9'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    environment:
      - MLFLOW_TRACKING_URI=http://host.docker.internal:5000
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.11.1
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    environment:
      - BACKEND_STORE_URI=sqlite:///mlflow.db
      - ARTIFACT_ROOT=./mlruns
    command: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0

# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.env
venv/
.git
.gitignore
mlruns/

# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
