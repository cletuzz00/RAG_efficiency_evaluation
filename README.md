Retrieval-Augmented Generation (RAG) – Retrieval Efficiency Benchmark
====================================================================

This repository contains a small, reproducible framework to compare **dense**, **sparse**, and **hybrid** retrieval for RAG-style systems over BEIR datasets (e.g. **NFCorpus**, FIQA), using a composite **Retrieval Efficiency Metric (REM)** that balances accuracy, latency, and cost. A single config file in `configs/experiment_config.yaml` drives data paths, Qdrant, and REM weights.

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

- `data/` – BEIR documents and queries CSVs (see `data/README.md`). Generated via `scripts/export_beir_to_csv.py`.
- `configs/` – single experiment config: `experiment_config.yaml`.
- `src/` – Python modules:
  - `dataset.py` – load documents and queries.
  - `embeddings.py` – SentenceTransformers embedding utilities.
  - `vector_store_qdrant.py` – Qdrant client wrapper.
  - `retrieval_dense.py` – dense retrieval (Qdrant).
  - `retrieval_sparse.py` – BM25-based sparse retrieval.
  - `retrieval_hybrid.py` – hybrid retrieval (dense + sparse).
  - `metrics.py` – accuracy/latency/cost + REM.
  - `runner.py` – experiment orchestration and logging.
  - `visualization.py` – plots from the logs.
- `logs/` – generated CSV logs (`logs_<dataset>.csv`) and per-dataset BM25 cache.
- `figures/` – saved bar and radar plots.
- `report/` – paper/report source (Markdown or LaTeX).

### 4. Basic Workflow

1. **Prepare BEIR data** (if not already done): download a BEIR dataset (e.g. NFCorpus or FIQA) into `data/beir/<dataset>/`, then export to project CSVs:

   ```bash
   # NFCorpus (default in config)
   python scripts/export_beir_to_csv.py --dataset nfcorpus --split test --beir-root data/beir --out-dir data

   # Or FIQA
   python scripts/export_beir_to_csv.py --dataset fiqa --split test --beir-root data/beir --out-dir data
   ```

   This produces `data/<dataset>_test_documents.csv` and `data/<dataset>_test_queries.csv` (with `relevant_doc_ids` for accuracy).

2. **Index documents**: from `src/`, embed documents and populate Qdrant (uses paths in `configs/experiment_config.yaml`):

   ```bash
   cd src && python index_documents.py
   ```

3. **Run experiments**: evaluate dense, sparse, and hybrid retrieval and compute REM; results are written to `logs/logs_<dataset>.csv` (e.g. `logs_nfcorpus.csv`):

   ```bash
   cd src && python runner.py
   ```

4. **Visualization**: generate comparison plots into `figures/` (e.g. from `src/` run the visualization script with the default config path).

**Switching datasets:** Set `data.dataset` to `nfcorpus` or `fiqa` in `configs/experiment_config.yaml`. Data paths, embedding cache, BM25 cache, and log file are then derived per dataset (e.g. `doc_embeddings_nfcorpus.npy`, `bm25_nfcorpus.pkl`, `logs_nfcorpus.csv`), so there is no cross-talk. After changing `dataset`, re-run `index_documents.py` then `runner.py`. Optionally, to remove the other dataset's cache and log files, run from the project root: `python scripts/clean_dataset_caches.py`.

**Faster runs:** In `configs/experiment_config.yaml` you can set `data.max_queries` and/or `data.max_documents` (e.g. 500 and 10000) to cap the test data; remove or comment out those keys for full-data runs.

### 5. Tunable evaluation (FIQA)

You can sweep retrieval and metric parameters on the FIQA dataset and aggregate results in one place. From the **project root**:

```bash
python scripts/run_tuning.py
```

- **Config:** Edit `configs/tuning_fiqa.yaml` to change the parameter grid (`retrieval.top_k`, `retrieval.hybrid.w_dense`, `metrics.accuracy_at_k`) and the base experiment config path. The script forces `data.dataset: fiqa` and runs the same evaluation runner once per combination.
- **Output:** Combined results are written to `logs/tuning_fiqa.csv` (one row per query × retrieval type × run). The script prints the best run by mean accuracy and the best run by mean REM, with their parameter combinations.

No changes to the core runner or retrievers are required; temp configs are written under `configs/tmp/` and removed after each run.

Refer to `configs/experiment_config.yaml` and the source files under `src/` for configuration details and extensibility.
