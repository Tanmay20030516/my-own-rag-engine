# my-own-rag-engine

A vector database implemented from scratch in Python (numpy only, no FAISS/sklearn/torch), plus a RAG pipeline on top of a local Ollama LLM. A learning project inspired by FAISS.

## Status

| Component | Status |
|---|---|
| `distance.py` — L2, cosine, inner product (scalar + pairwise) | done |
| `kmeans.py` — Lloyd's k-means | done |
| `index/base.py` — `VectorIndex` interface | done |
| `index/flat.py` — exact brute-force search | done |
| `index/ivf.py` — cluster + probe ANN | done |
| `index/pq.py` — product quantization ANN | done |
| `index/hnsw.py` — graph-based ANN | done |
| `storage.py` — save/load | done (pickle, via `VectorIndex.save/load`) |
| RAG pipeline (`rag/`) | not started |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pytest
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

## Tests

```bash
pytest tests/
```
