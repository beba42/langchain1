# LangChain Chat Stack

This top-level project bundles two containers:

- `chat` — the LangChain Gradio application that serves the local document QA agent.
- `ollama` — an Ollama server preloaded with `llama3.1:8b` and `nomic-embed-text`.

## Layout

- `chat/` — original Python project (see `chat/README.md` for details).
- `ollama/` — Dockerfile that builds an Ollama image with required models.
- `docker-compose.yml` — defines the multi-container stack.
- `data/` — shared volume mounted into the chat container for document ingestion.

## Prerequisites

- Docker 24+ (or a recent Docker Desktop).
- Enough disk space for Ollama models (~15 GB for the selected trio).

## Usage

1. **Build and start the stack**
   ```bash
   docker compose up --build
   ```
   The first run downloads the Ollama base image and pulls the models; expect several minutes.

2. **Access the app**  
   Visit `http://localhost:7860` to reach the Gradio UI.

3. **Add documents**  
   Drop `.txt` files into the top-level `data/` directory on the host. The folder is mounted at `/data` inside the chat container (and mapped to `LC_DATA_DIR`), so new files are available on restart.

4. **Stop the stack**
   ```bash
   docker compose down
   ```

### Updating Models or Requirements

- To pull additional Ollama models, add `ollama pull <model>` lines to `ollama/Dockerfile`, then rebuild with `docker compose build ollama`.
- To change Python dependencies or app code, edit files under `chat/` and rebuild `docker compose build chat`.

### Data Persistence

- `data/` (mounted into `/data`) holds your knowledge base documents.
- `ollama-data` named volume keeps Ollama’s downloaded models so they persist between container restarts.

## Troubleshooting

- If the chat service starts before Ollama finishes booting, `docker compose up` automatically restarts it once Ollama is ready.
- Port conflicts: adjust the `ports` section in `docker-compose.yml`.
