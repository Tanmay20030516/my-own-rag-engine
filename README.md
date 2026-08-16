# VecLite

A vector database written from scratch in numpy — Flat, IVF, PQ and HNSW behind one
interface — with a local RAG pipeline and a Streamlit chat UI on top of it. No faiss,
no sklearn, no torch in the engine, and no data leaves your machine.

## Features

- **Four index types** behind a common `VectorIndex` interface (`train` → `add` →
  `search`), each supporting `l2`, `cosine` and `ip` metrics, with pickle-backed
  `save()`/`load()`.
- **Shared primitives**: vectorized distance kernels and Lloyd's k-means, both used
  internally by IVF and PQ.
- **A RAG pipeline** — three chunking strategies (fixed-window, sentence-window, and
  semantic splitting on embedding drift), a `nomic-embed-text` embedder that applies the
  correct task prefixes, and retrieval-grounded answers from a local LLM.
- **A Streamlit chat UI**: upload documents, index them with any of the four indexes,
  and get cited answers streamed back.
- **One-command deployment** — `docker compose up --build` brings up Ollama and the UI
  together.
- **A full benchmark against faiss**: 79 configurations on real sentence embeddings,
  written up in [`report.md`](report.md) / [`report.pdf`](report.pdf) with six figures.
- **125 tests**, none of which need a live Ollama server.

## Layout


| Path                                           | Contents                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------ |
| [`vectordb/distance.py`](vectordb/distance.py) | L2, cosine, inner product — scalar and pairwise, all "lower = closer"       |
| [`vectordb/kmeans.py`](vectordb/kmeans.py)     | Lloyd's k-means with k-means++ init                                          |
| [`vectordb/index/`](vectordb/index/)           | `base.py` (interface + persistence), `flat.py`, `ivf.py`, `pq.py`, `hnsw.py` |
| [`rag/`](rag/)                                 | `chunker.py`, `embedder.py`, `pipeline.py`                                   |
| [`frontend/`](frontend/)                       | Streamlit app, index-building service, document loaders                      |
| [`examples/`](examples/)                       | Benchmarks, hyperparameter sweep, figure generation, RAG demo                |
| [`tests/`](tests/)                             | pytest suite                                                                 |

The engine has no dependency on the frontend, and the frontend imports `rag` and
`vectordb` directly rather than over HTTP.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r py-requirements.txt
```

faiss and matplotlib are in there for the benchmarks only — the `vectordb` package
itself needs nothing but numpy.

The RAG pipeline needs a local [Ollama](https://ollama.com) server:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

## Usage

### Vector indexes

```python
from vectordb.index import FlatIndex, IVFIndex, PQIndex, HNSWIndex

index = FlatIndex(metric="l2")          # exact brute force
index.add(vectors, ids)
distances, ids = index.search(query, k=5)

index = IVFIndex(nlist=100, nprobe=10)  # cluster and probe
index.train(vectors)
index.add(vectors, ids)

index = PQIndex(M=8, Ks=256)            # product quantization
index.train(vectors)
index.add(vectors, ids)

index = HNSWIndex(M=16, ef_construction=200, ef_search=50)  # graph
index.add(vectors, ids)

index.save("index.pkl")
index = HNSWIndex.load("index.pkl")
```

IVF and PQ need a `train()` pass to fit centroids and codebooks; Flat and HNSW treat it
as a no-op. `search()` always returns `(distances, ids)`, sorted nearest-first.

### RAG pipeline

Embeddings and generation both run on local Ollama:

```python
from rag.chunker import chunk_sentences
from rag.embedder import OllamaEmbedder
from rag.pipeline import RAGPipeline
from vectordb.index import FlatIndex

pipeline = RAGPipeline(
    index=FlatIndex(metric="cosine"),
    embedder=OllamaEmbedder(model="nomic-embed-text"),
    chunker=chunk_sentences,
    llm_model="qwen3:8b",
)
pipeline.index_documents([document_text])

hits = pipeline.retrieve("What does an IVF index trade off for speed?", k=3)
answer = pipeline.query("What does an IVF index trade off for speed?", k=3)
```

`retrieve()` returns the matching chunks with their distances if you want to handle
generation yourself; `query()` does the whole round trip. A runnable version:

```bash
PYTHONPATH=. python examples/rag_demo.py
```

### Web UI

A chat interface over the engine: upload `.txt`, `.md`, `.pdf` or `.docx` files, pick an
index type and chunking strategy, and ask questions. Answers stream in with inline
citations, and an expander shows the passages they were grounded in along with their
distances. Follow-ups like *"what about the second one?"* are rewritten into standalone
queries before retrieval, and the rewritten query is displayed so retrieval stays
inspectable.

```bash
docker compose up --build     # then open http://localhost:8501
```

Compose starts three services: Ollama, a one-shot job that pulls the embedding and chat
models, and the UI, which waits on that pull so the first question can't fail on a
missing model. **The first run downloads ~5 GB of model weights**, cached in the
`ollama-models` volume — later starts are fast.

Already running Ollama on your host? Skip the bundled one:

```bash
OLLAMA_HOST=http://host.docker.internal:11434 \
  docker compose up rag-ui --build --no-deps
```

To run without Docker, install the UI's dependencies and launch it from the repo root:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

`OLLAMA_HOST`, `RAG_EMBED_MODEL` and `RAG_LLM_MODEL` configure it either way, and are
also editable at runtime in the sidebar. See [`frontend/README.md`](frontend/README.md)
for details.

## Benchmarking

[`examples/real_search.py`](examples/real_search.py) builds all four indexes over real
sentence embeddings and reports recall@10 (against `FlatIndex`'s exact search as ground
truth), build time, search latency and memory — then repeats IVF/PQ/HNSW on faiss with
matching hyperparameters. It downloads *Pride and Prejudice* from Project Gutenberg,
chunks it, embeds it with the project's own `OllamaEmbedder`, and scores against
hand-written questions about the plot. It also prints the retrieved text for one query,
so results can be eyeballed rather than trusted on a number.

```bash
PYTHONPATH=.:examples python examples/real_search.py
```

Example run — 1,981 chunks, cosine metric, faiss side normalized + inner product;
numbers vary by run, hyperparameters and hardware:

```
Ours     Recall@10  Build(s)  Search(ms)  Memory(MB)
Flat        1.000     0.000       1.410       11.6
IVF         0.925     0.102       0.514       11.9
PQ          0.400     0.143       0.161        0.4
HNSW        1.000     4.306       0.879       13.8

faiss    Recall@10  Build(s)  Search(ms)  Memory(MB)
Flat        1.000     0.002       0.362        5.8
IVF         0.933     0.006       0.054        5.9
PQ          0.308     0.037       0.044        0.2
HNSW        1.000     0.035       0.046        6.1
```

IVF matches faiss's recall to within 0.01, and HNSW hits exact recall on both sides.
Both PQ implementations trail badly at `M=8, Ks=64` — not enough codebook resolution for
768 dimensions, which is a property of the method at those settings rather than of
either implementation. Memory sits at ~2× faiss throughout: float64 against float32,
with no structural waste. HNSW's build time is the one structural gap — an incremental
per-node graph build doesn't vectorize the way k-means training does, so it scales like
a Python loop rather than like C++.

[`examples/basic_search.py`](examples/basic_search.py) runs the same protocol on 100k
random 512-dim vectors. Random Gaussians have no structure for an approximate index to
exploit, so every method scores equally badly there — treat it as a build-time and
memory stress test, not a recall signal.

## Report

[`report.md`](report.md) / [`report.pdf`](report.pdf) is the full write-up: 79
configurations sweeping `nprobe`/`nlist`, `M`/`Ks` and `M`/`ef_construction`/`ef_search`,
ours against faiss, with detailed tables and six figures under [`plots/`](plots/).

Highlights:

- Every configuration that should be exact scores precisely 1.000 on both sides, and
  recall tracks faiss within 0.02 almost everywhere.
- For PQ, `M=32, Ks=64` beats `M=8, Ks=256` on recall **and** memory simultaneously
  (0.440 vs 0.328, at 0.58 MB vs 1.58 MB) — if you tune one PQ parameter, tune `M`.
- Search runs 15–24× slower than faiss and HNSW builds ~130× slower; only the HNSW build
  gap is structural, the rest is a constant factor from Python and float64.

Every table and figure in the report is generated, not transcribed:

```bash
PYTHONPATH=.:examples python examples/sweep.py    # -> sweep_results.json
PYTHONPATH=.:examples python examples/plots.py    # -> plots/*.png, plots/tables.md
```

Rebuilding the PDF needs a LaTeX install providing `xelatex`:

```bash
python -c "import pypandoc; pypandoc.convert_file('report.md','pdf',outputfile='report.pdf',\
extra_args=['--pdf-engine=xelatex','--variable=geometry:margin=1.6cm','--toc'])"
```

## Tests

```bash
pytest tests/
```

125 tests covering the distance kernels, k-means, all four indexes (exactness,
persistence, edge cases), the chunkers, the embedder's prefixing, pipeline retrieval,
and the frontend's streaming filter and index-sizing rules. Ollama is mocked throughout,
so the suite runs offline in under a second.
