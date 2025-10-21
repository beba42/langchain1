# syntax=docker/dockerfile:1

FROM ollama/ollama:latest

SHELL ["bash", "-lc"]

# Pre-pull required models during build so the container is ready at runtime.
RUN ollama serve & \
    pid=$! && \
    sleep 5 && \
    ollama pull llama3.1:8b && \
    ollama pull nomic-embed-text && \
    kill $pid && \
    wait $pid || true

EXPOSE 11434
