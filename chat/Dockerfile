# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    LC_DATA_DIR=/data

WORKDIR /app

# System deps for building Python wheels and talking to host-side Ollama if needed
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data && \
    cp -r langchain_app/data/. /data/

EXPOSE 7860
CMD ["python", "scripts/run_local_agent.py"]
