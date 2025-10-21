# LangChain Local/Hybrid Agent Starter

This repository scaffolds a LangChain project that combines local document retrieval with optional web search.

## Layout

- `langchain_app/` — Python package that contains agents, tools, and configuration.
- `langchain_app/data/` — Seed documents for local ingestion (override with `LC_DATA_DIR` when containerised).
- `langchain_app/agents/` — Local document utilities and a DuckDuckGo-based web search agent.
- `langchain_app/config/` — Centralised settings powered by Pydantic.
- `langchain_app/tools/` — Placeholder for custom LangChain tools you add later.
- `scripts/` — Runnable entry points (e.g. `run_local_agent.py`).
- `tests/` — Pytest-based test suite scaffolding.

## Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Provide credentials**  
   Copy `.env.example` to `.env` and fill in the required API keys (e.g. `OPENAI_API_KEY`).
3. **Add local documents**  
   Drop `.txt` files into `langchain_app/data/` or set the `LC_DATA_DIR` environment variable to point elsewhere.
4. **Run the local agent**
   ```bash
   python scripts/run_local_agent.py
   ```

## Docker

From the repository root you can run the multi-service stack (`ollama` + `chat`) via:
```bash
docker compose up --build
```
If you prefer to build only this service for standalone use, run the following from inside `chat/`:
```bash
docker build -t langchain-local-agent .
docker run --rm -it \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e LC_DATA_DIR=/data \
  -p 7860:7860 \
  -v $(pwd)/../data:/data \
  langchain-local-agent
```
Adjust the `OLLAMA_HOST` value if your Ollama server uses a different address.

## Next Steps

- Connect additional LangChain tools (browsing, code execution, etc.) under `langchain_app/tools`.
- Wire the agents into LangGraph or LangServe for more advanced orchestration.
- Add CI/CD workflows before publishing to GitHub.
