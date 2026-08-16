# RAG Engine UI

A Streamlit chat app over the engine in this repo: upload documents, index them
with one of the from-scratch vector indexes in [`vectordb/`](../vectordb/), and
ask questions answered by a local Ollama LLM grounded in the retrieved chunks.

Nothing leaves your machine — embeddings and generation both run on local Ollama.

## Run with Docker (recommended)

Brings up Ollama and the UI together, and pulls the models on first start:

```bash
docker compose up --build
# then open http://localhost:8501
```

The first run downloads ~5 GB of weights into the `ollama-models` volume; they
persist, so later starts are fast.

Already have Ollama running on your host? Skip the bundled one:

```bash
OLLAMA_HOST=http://host.docker.internal:11434 \
  docker compose up rag-ui --build --no-deps
```

## Run locally

```bash
pip install -r py-requirements.txt -r frontend/requirements.txt
ollama serve
ollama pull nomic-embed-text
ollama pull qwen3:8b

streamlit run frontend/app.py     # from the repo root
```

Run it from the repo root — the app imports `rag` and `vectordb` from there, and
Streamlit reads [`.streamlit/config.toml`](../.streamlit/config.toml) out of the
working directory.

## Using it

1. **Upload** one or more `.txt`, `.md`, `.pdf`, or `.docx` files in the sidebar.
2. Pick an **index type** and **chunking** strategy, then hit **Build index**.
3. **Ask questions** in the chat. Every answer cites the passages it used, and
   the expander under it shows those passages with their distances.

Changing the documents or the chunking settings invalidates the index; the app
tells you to rebuild rather than silently answering from stale vectors.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server the app talks to |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_LLM_MODEL` | `qwen3:8b` | Chat model |

All three are also editable at runtime under **Retrieval & models** in the sidebar.

## Layout

| File | Role |
| --- | --- |
| [`app.py`](app.py) | Streamlit UI: sidebar, ingest, chat loop |
| [`rag_service.py`](rag_service.py) | Index construction, question condensing, streaming |
| [`loaders.py`](loaders.py) | File bytes → text (txt/md/pdf/docx) |

## Notes on the design

**Index sizing.** IVF and PQ fit centroids over the corpus, so their parameters
can't exceed what the data supports — k-means needs at least one vector per
cluster. `rag_service` clamps `nlist`/`Ks` to the chunk count and surfaces a
warning instead of failing on a short upload. At that scale their recall and
compression numbers aren't meaningful; they're there to make the tradeoffs
visible, not to benchmark them. Use [`examples/sweep.py`](../examples/sweep.py)
for that.

**Follow-up questions.** "What about the second one?" embeds to nothing useful on
its own. When the toggle is on, the app first asks the LLM to rewrite the message
into a standalone query using the recent transcript, then retrieves with that.
The rewritten query is shown above the answer so retrieval stays inspectable.

**Reasoning traces.** qwen3 emits `<think>...</think>` blocks. Those are filtered
out of the token stream as it arrives rather than by passing `think=False` to
Ollama, which some models reject outright.
