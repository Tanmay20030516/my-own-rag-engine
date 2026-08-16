# Streamlit RAG UI. Build context is the repo root because the app imports
# the engine packages (rag/, vectordb/) directly rather than over HTTP.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so dependency layers survive edits to application code.
COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

COPY rag/ rag/
COPY vectordb/ vectordb/
COPY frontend/ frontend/
COPY .streamlit/ .streamlit/

# Ollama runs outside this container. docker-compose points this at the
# `ollama` service; override it to reach a server on the host instead.
ENV OLLAMA_HOST=http://ollama:11434 \
    RAG_EMBED_MODEL=nomic-embed-text \
    RAG_LLM_MODEL=qwen3:8b

# Streamlit writes to ~/.streamlit at runtime, so the app must not run as root
# with a non-writable home.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
