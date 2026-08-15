# my-own-rag-engine

A vector database implemented from scratch in Python (numpy only, no FAISS/sklearn/torch), plus a RAG pipeline on top of a local Ollama LLM. A learning project inspired by FAISS.

## Status


| Component                                                                             | Status                                    |
| --------------------------------------------------------------------------------------- | ------------------------------------------- |
| `distance.py` — L2, cosine, inner product (scalar + pairwise)                        | done                                      |
| `kmeans.py` — Lloyd's k-means                                                        | done                                      |
| `index/base.py` — `VectorIndex` interface                                            | done                                      |
| `index/flat.py` — exact brute-force search                                           | done                                      |
| `index/ivf.py` — cluster + probe ANN                                                 | done                                      |
| `index/pq.py` — product quantization ANN                                             | done                                      |
| `index/hnsw.py` — graph-based ANN                                                    | done                                      |
| `storage.py` — save/load                                                             | done (pickle, via`VectorIndex.save/load`) |
| `rag/embedder.py` — `OllamaEmbedder` (nomic-embed-text)                              | done                                      |
| `rag/chunker.py` — fixed, sentence, and semantic chunking                            | done                                      |
| `rag/pipeline.py` — `RAGPipeline` (index docs → query → local LLM)                 | done                                      |
| `examples/basic_search.py` — benchmark vs. faiss (recall, build/search time, memory) | done                                      |
| `examples/real_search.py` — same benchmark on real sentence embeddings             | done                                      |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r py-requirements.txt
```

RAG pipeline also needs a local [Ollama](https://ollama.com) server:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

## Usage

```python
from vectordb.index import FlatIndex, IVFIndex, PQIndex, HNSWIndex

index = FlatIndex(metric="l2")
index.add(vectors, ids)
distances, ids = index.search(query, k=5)

# IVF and PQ need a train() pass before add(); HNSW and Flat don't
index = IVFIndex(nlist=100, nprobe=10)
index.train(vectors)
index.add(vectors, ids)

index = PQIndex(M=8, Ks=256)
index.train(vectors)
index.add(vectors, ids)

index = HNSWIndex(M=16, ef_construction=200, ef_search=50)
index.add(vectors, ids)
```

RAG pipeline, fully local via Ollama for both embeddings and generation:

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
answer = pipeline.query("What does an IVF index trade off for speed?", k=3)
```

See `examples/rag_demo.py` for a runnable version:

```bash
PYTHONPATH=. python examples/rag_demo.py
```

## Benchmarking

`examples/basic_search.py` builds all four index types on the same random
dataset and reports recall@k (against `FlatIndex`'s exact search as ground
truth), build time, search latency, and memory footprint -- then repeats
IVF/PQ/HNSW against [faiss](https://github.com/facebookresearch/faiss) with matching hyperparameters, so our from-scratch implementations have a reference point. faiss is only needed for this comparison, not for the core `vectordb` package.

```bash
PYTHONPATH=. python examples/basic_search.py
```

Random Gaussian vectors have no exploitable structure, though, so every
approximate index looks equally bad on them regardless of implementation
quality -- it's a build-time/memory stress test, not a recall signal.
`examples/real_search.py` runs the same protocol on real sentence embeddings
instead: it downloads *Pride and Prejudice* from Project Gutenberg, chunks
it, embeds the chunks with the project's own `OllamaEmbedder`
(nomic-embed-text), and scores recall against hand-written questions about
the book's plot. It also prints the actual retrieved text for one query, so
you can eyeball whether the results make sense rather than trust a number.
Requires `ollama serve` running with `nomic-embed-text` pulled (see Setup).

```bash
PYTHONPATH=. python examples/real_search.py
```

Example run (1,981 chunks, cosine metric, faiss side normalized + inner
product -- numbers will vary by run, hyperparameters, and hardware):

```
Ours     Recall@10  Build(s)  Search(ms)  Memory(MB)
Flat        1.000     0.000       1.693       11.6
IVF         0.950     0.112       0.557       11.9
PQ          0.308     0.094       0.153        0.2
HNSW        1.000     4.356       0.938       13.8

faiss    Recall@10  Build(s)  Search(ms)  Memory(MB)
Flat        1.000     0.001       0.295        5.8
IVF         0.933     0.006       0.053        5.9
PQ          0.267     0.029       0.039        0.1
HNSW        1.000     0.040       0.049        6.1
```

IVF and HNSW essentially match faiss's recall here (HNSW hits exact recall
on both); PQ trails on both sides at `M=8, Ks=32` -- expected, since that's
not enough codebook resolution to preserve fine distinctions at 768
dimensions, and it's the same gap on faiss's own PQ, not an implementation
defect. HNSW's build time is the one place with a structural, not just
tunable, gap vs. faiss: it's a pure per-node incremental graph build, which
doesn't vectorize the way k-means-based training does, so it scales the way
a pure-Python loop scales rather than the way a C++ implementation does.

## Tests

```bash
pytest tests/
```
