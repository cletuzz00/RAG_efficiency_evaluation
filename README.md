Retrieval-Augmented Generation (RAG) – Retrieval Efficiency Benchmark
====================================================================

This repository contains a small, reproducible framework to compare **dense**, **sparse**, and **hybrid** retrieval for RAG-style systems over business documents, using a composite **Retrieval Efficiency Metric (REM)** that balances accuracy, latency, and cost.

### 1. Environment Setup

- Ensure you have **Python 3.10+** installed.
- (Optional but recommended) create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # on macOS/Linux
venv\\Scripts\\activate   # on Windows
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant with Podman

From the repository root:

```bash
podman run -d \
  --name qdrant \
  -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Verify Qdrant is running:

```bash
curl http://localhost:6333/collections
```

You should get a JSON response (possibly an empty list of collections).

### 3. Project Layout

- `data/` – business documents and query–answer CSVs.
- `configs/` – experiment configuration, including REM weights.
- `src/` – Python modules:
  - `dataset.py` – load/prepare documents and queries.
  - `embeddings.py` – SentenceTransformers embedding utilities.
  - `vector_store_qdrant.py` – Qdrant client wrapper.
  - `retrieval_dense.py` – dense retrieval (Qdrant).
  - `retrieval_sparse.py` – BM25-based sparse retrieval.
  - `retrieval_hybrid.py` – hybrid retrieval (dense + sparse).
  - `metrics.py` – accuracy/latency/cost + REM.
  - `runner.py` – experiment orchestration and logging.
  - `visualization.py` – plots from the logs.
- `logs/` – generated CSV logs (e.g. `logs_all.csv`).
- `figures/` – saved bar and radar plots.
- `report/` – paper/report source (Markdown or LaTeX).

### 4. Basic Workflow

1. Place or generate your business document corpus in `data/business_corpus.csv` and query–answer pairs in `data/queries_answers.csv`.
2. Run the indexing script (from `src/`) to embed documents and populate Qdrant.
3. Run the experiment runner to evaluate dense, sparse, and hybrid retrieval and compute REM; results are written to `logs/logs_all.csv`.
4. Use the visualization script to generate comparison plots into `figures/`.Refer to the source files under `src/` for configuration details and extensibility.