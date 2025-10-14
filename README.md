# LangChain Local/Hybrid Agent Starter

This repository scaffolds a LangChain project that combines local document retrieval with optional web search.

## Layout

- `langchain_app/` — Python package that contains agents, tools, and configuration.
- `langchain_app/data/` — Drop plain-text files here; the local agent will ingest them automatically.
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
   Drop `.txt` files into `langchain_app/data/`.
4. **Run the local agent**
   ```bash
   python scripts/run_local_agent.py
   ```

## Next Steps

- Connect additional LangChain tools (browsing, code execution, etc.) under `langchain_app/tools`.
- Wire the agents into LangGraph or LangServe for more advanced orchestration.
- Add CI/CD workflows before publishing to GitHub.

