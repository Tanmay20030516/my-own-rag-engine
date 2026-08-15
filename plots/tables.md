### Headline: all four indexes at default hyperparameters

| Index | recall@10 (ours) | recall@10 (faiss) | build s (ours) | build s (faiss) | search ms (ours) | search ms (faiss) | memory MB (ours) | memory MB (faiss) |
|---|---|---|---|---|---|---|---|---|
| Flat (exact) | 1.000 | 1.000 | 0.00 | 0.00 | 3.282 | 0.218 | 31.46 | 15.71 |
| IVF (nlist=64, nprobe=8) | 0.838 | 0.845 | 0.41 | 0.02 | 0.976 | 0.041 | 32.03 | 15.94 |
| PQ (M=8, Ks=64) | 0.208 | 0.207 | 0.58 | 0.10 | 0.322 | 0.503 | 0.46 | 0.22 |
| HNSW (M=16, efC=200, efS=50) | 0.968 | 0.987 | 20.50 | 0.16 | 1.411 | 0.061 | 37.22 | 16.45 |

### IVF — nprobe sweep (nlist=64)

Only search changes, so build time is the one shared build.

| nprobe | % scanned | recall@10 (ours) | recall@10 (faiss) | build s (ours) | build s (faiss) | search ms (ours) | search ms (faiss) | memory MB (ours) | memory MB (faiss) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.6% | 0.469 | 0.486 | 0.41 | 0.02 | 0.137 | 0.014 | 32.03 | 15.94 |
| 2 | 3.1% | 0.623 | 0.630 | 0.41 | 0.02 | 0.262 | 0.015 | 32.03 | 15.94 |
| 4 | 6.2% | 0.736 | 0.750 | 0.41 | 0.02 | 0.520 | 0.025 | 32.03 | 15.94 |
| 8 | 12.5% | 0.838 | 0.845 | 0.41 | 0.02 | 0.976 | 0.041 | 32.03 | 15.94 |
| 16 | 25.0% | 0.921 | 0.918 | 0.41 | 0.02 | 1.581 | 0.073 | 32.03 | 15.94 |
| 32 | 50.0% | 0.974 | 0.974 | 0.41 | 0.02 | 4.347 | 0.132 | 32.03 | 15.94 |
| 64 | 100.0% | 1.000 | 1.000 | 0.41 | 0.02 | 7.557 | 0.231 | 32.03 | 15.94 |

### IVF — nlist sweep

nprobe fixed at 8; each row is a fresh k-means training run.

| nlist | recall@10 (ours) | recall@10 (faiss) | build s (ours) | build s (faiss) | search ms (ours) | search ms (faiss) | memory MB (ours) | memory MB (faiss) |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.956 | 0.952 | 0.36 | 0.01 | 4.485 | 0.128 | 31.75 | 15.80 |
| 32 | 0.880 | 0.891 | 0.41 | 0.01 | 1.553 | 0.074 | 31.84 | 15.85 |
| 64 | 0.838 | 0.845 | 0.40 | 0.01 | 0.986 | 0.041 | 32.03 | 15.94 |
| 128 | 0.805 | 0.795 | 0.77 | 0.02 | 0.571 | 0.029 | 32.42 | 16.13 |
| 256 | 0.753 | 0.735 | 1.23 | 0.03 | 0.438 | 0.028 | 33.19 | 16.50 |

### PQ — Ks sweep (M=8)

5363 training vectors, so training points per centroid = 5363/Ks.

| Ks | train pts / centroid | recall@10 (ours) | recall@10 (faiss) | build s (ours) | build s (faiss) | search ms (ours) | search ms (faiss) | memory MB (ours) | memory MB (faiss) |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 335 | 0.123 | 0.121 | 0.28 | 0.05 | 0.261 | 0.489 | 0.18 | 0.07 |
| 32 | 168 | 0.166 | 0.154 | 0.41 | 0.09 | 0.289 | 0.493 | 0.27 | 0.12 |
| 64 | 84 | 0.208 | 0.207 | 0.58 | 0.10 | 0.322 | 0.503 | 0.46 | 0.22 |
| 128 | 42 | 0.274 | 0.264 | 1.00 | 0.16 | 0.344 | 0.505 | 0.83 | 0.41 |
| 256 | 21 | 0.328 | 0.322 | 1.69 | 0.30 | 0.363 | 0.476 | 1.58 | 0.79 |

### PQ — M sweep (Ks=64)

Each vector is stored as M codes; more subspaces = finer approximation, more bytes.

| M | dims / subspace | recall@10 (ours) | recall@10 (faiss) | build s (ours) | build s (faiss) | search ms (ours) | search ms (faiss) | memory MB (ours) | memory MB (faiss) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 192 | 0.148 | 0.124 | 0.69 | 0.06 | 0.210 | 0.261 | 0.44 | 0.20 |
| 8 | 96 | 0.208 | 0.207 | 0.58 | 0.11 | 0.303 | 0.507 | 0.46 | 0.22 |
| 16 | 48 | 0.301 | 0.293 | 0.69 | 0.20 | 0.509 | 0.999 | 0.50 | 0.25 |
| 32 | 24 | 0.440 | 0.426 | 0.76 | 0.54 | 0.859 | 1.939 | 0.58 | 0.31 |

### PQ — L2 vs inner-product ADC table (faiss)

Identical codebooks and identical normalized vectors; only the distance table used at search time differs. Our PQIndex normalizes and then uses L2 per subspace, so the L2 column is the apples-to-apples comparison.

| M | Ks | recall (ours, L2) | recall (faiss, L2) | recall (faiss, inner product) | IP penalty |
|---|---|---|---|---|---|
| 8 | 16 | 0.123 | 0.121 | 0.085 | +0.036 |
| 8 | 32 | 0.166 | 0.154 | 0.121 | +0.033 |
| 8 | 64 | 0.208 | 0.207 | 0.155 | +0.052 |
| 8 | 128 | 0.274 | 0.264 | 0.236 | +0.028 |
| 8 | 256 | 0.328 | 0.322 | 0.280 | +0.042 |
| 4 | 64 | 0.148 | 0.124 | 0.112 | +0.012 |
| 8 | 64 | 0.208 | 0.207 | 0.155 | +0.052 |
| 16 | 64 | 0.301 | 0.293 | 0.244 | +0.050 |
| 32 | 64 | 0.440 | 0.426 | 0.357 | +0.069 |

### HNSW — ef_search sweep (M=16, ef_construction=200)

Search-time only, so build time is the one shared build.

| ef_search | recall@10 (ours) | recall@10 (faiss) | build s (ours) | build s (faiss) | search ms (ours) | search ms (faiss) | memory MB (ours) | memory MB (faiss) |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.836 | 0.831 | 20.50 | 0.16 | 0.438 | 0.029 | 37.22 | 16.45 |
| 20 | 0.910 | 0.926 | 20.50 | 0.16 | 0.800 | 0.033 | 37.22 | 16.45 |
| 50 | 0.968 | 0.987 | 20.50 | 0.16 | 1.411 | 0.061 | 37.22 | 16.45 |
| 100 | 0.990 | 0.997 | 20.50 | 0.16 | 2.519 | 0.104 | 37.22 | 16.45 |
| 200 | 0.998 | 1.000 | 20.50 | 0.16 | 4.288 | 0.183 | 37.22 | 16.45 |
| 400 | 0.999 | 1.000 | 20.50 | 0.16 | 7.177 | 0.330 | 37.22 | 16.45 |

### HNSW — M sweep

ef_construction=200, ef_search=50.

| M | recall@10 (ours) | recall@10 (faiss) | build s (ours) | build s (faiss) | search ms (ours) | search ms (faiss) | memory MB (ours) | memory MB (faiss) |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.903 | 0.947 | 14.79 | 0.13 | 1.098 | 0.039 | 35.05 | 16.12 |
| 16 | 0.968 | 0.986 | 23.82 | 0.16 | 1.570 | 0.056 | 37.22 | 16.45 |
| 32 | 0.992 | 0.992 | 31.58 | 0.16 | 2.410 | 0.063 | 41.70 | 17.10 |

### HNSW — ef_construction sweep

M=16, ef_search=50. efC = ef_construction.

| efC | recall@10 (ours) | recall@10 (faiss) | build s (ours) | build s (faiss) | search ms (ours) | search ms (faiss) | memory MB (ours) | memory MB (faiss) |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.961 | 0.965 | 9.01 | 0.06 | 1.206 | 0.047 | 37.20 | 16.45 |
| 100 | 0.967 | 0.981 | 14.40 | 0.10 | 1.187 | 0.055 | 37.21 | 16.45 |
| 200 | 0.968 | 0.984 | 16.87 | 0.15 | 1.066 | 0.054 | 37.22 | 16.45 |
| 400 | 0.969 | 0.984 | 25.06 | 0.25 | 1.256 | 0.057 | 37.22 | 16.45 |
