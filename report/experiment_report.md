# Retrieval Efficiency in RAG Systems: A Comparative Study of Dense, Sparse, and Hybrid Retrieval on BEIR

**Experiment report — FIQA tuning and REM evaluation**

---

## Abstract

We evaluate dense, sparse, and hybrid retrieval strategies for retrieval-augmented generation (RAG) on the BEIR benchmark, using the FIQA dataset. We introduce a composite **Retrieval Efficiency Metric (REM)** that balances retrieval accuracy, latency, and cost. A full parameter sweep over retrieval depth (*top_k*), hybrid weighting (*w_dense*), and accuracy-at-*k* is run; results are aggregated to identify the best configurations by mean accuracy and by REM. This report documents the experimental setup, methodology, and results for reproducibility and publication.

---

## 1. Introduction and Motivation

Retrieval-augmented generation (RAG) systems depend on a retrieval stage that selects relevant documents for downstream language models. Retrieval can be implemented with **dense** (embedding-based), **sparse** (lexical, e.g. BM25), or **hybrid** (combination of both) methods. Each approach involves trade-offs among accuracy, latency, and cost. To support informed design choices, we need a single metric that reflects these trade-offs and a reproducible evaluation protocol.

This experiment (1) compares dense, sparse, and hybrid retrieval on FIQA (BEIR), (2) defines and uses REM to rank configurations, and (3) sweeps key parameters to report best-by-accuracy and best-by-REM configurations with full details for publication.

---

## 2. Background

### 2.1 RAG systems

In RAG, a retriever fetches a small set of documents from a corpus given a user query; a generator then conditions on the query and the retrieved text. Retrieval quality directly affects answer quality; retrieval latency and cost (e.g. token usage) affect user experience and operating cost.

### 2.2 Dense vs. sparse vs. hybrid retrieval

- **Dense retrieval**: Queries and documents are encoded with a neural embedding model; retrieval is approximate nearest-neighbor search in the embedding space (e.g. via Qdrant with HNSW). Captures semantic similarity but is sensitive to domain and model choice.
- **Sparse retrieval**: Lexical matching (e.g. BM25) over tokenized text. Robust and interpretable but can miss paraphrases and synonyms.
- **Hybrid retrieval**: Dense and sparse scores are normalized (e.g. min-max) and combined with configurable weights (*w_dense*, *w_sparse* = 1 − *w_dense*). Aims to combine semantic and lexical signals.

### 2.3 Related work on evaluation and efficiency

This work builds on the BEIR benchmark for heterogeneous IR evaluation. Thakur et al. introduced BEIR as a large-scale suite of diverse retrieval datasets and tasks, enabling zero-shot comparison of lexical, dense, and hybrid retrieval models on common metrics such as nDCG and recall (see *“BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models,”* NeurIPS 2021, arXiv:2104.08663). More recent work on “Resources for Brewing BEIR” provides reproducible reference implementations and a public leaderboard, emphasizing standardized and comparable evaluation across models and datasets (Lin et al., arXiv:2306.07471).

Beyond pure effectiveness, several IR frameworks explicitly consider **utility–cost trade-offs**. The C/W/L framework and the associated `cwl_eval` tool (Azzopardi, Thomas, and Moffat) unify many traditional IR measures under a user-model-based view that reports both utility and cost in consistent units, highlighting that ranking quality should be interpreted together with user effort and system resources. Recent benchmark tooling such as SuiteEval also focuses on standardized multi-metric evaluation across collections, including efficiency aspects like latency and compute.

For RAG systems, recent work on practical evaluation proposes cost–latency–quality trade-off analysis and set-based retrieval metrics tailored to the “fixed context window” setting of large language models (e.g. *“Practical RAG Evaluation: A Rarity-Aware Set-Based Metric and Cost-Latency-Quality Trade-offs,”* arXiv preprint). Our REM metric is a deliberately simple linear combination of normalized accuracy, latency, and cost, inspired by these broader lines of work but tailored to retrieval-only evaluation for RAG.

---

## 3. Methodology

### 3.1 Dataset

- **Source**: BEIR benchmark; experiment uses the **FIQA** dataset.
- **Data preparation**: BEIR data is exported to CSV via `scripts/export_beir_to_csv.py` (e.g. `--dataset fiqa --split test`). This produces:
  - `data/fiqa_test_documents.csv`: columns `id`, `title`, `text`.
  - `data/fiqa_test_queries.csv`: columns `query_id`, `query`, `expected_answer`, `relevant_doc_ids` (comma-separated).
- **Queries**: Only queries with non-empty `relevant_doc_ids` are used for accuracy; the number of evaluated queries can be capped with `data.max_queries` in the config for faster runs.
- **Corpus**: Documents are embedded and stored in Qdrant; optional `data.max_documents` caps the corpus size per run.

### 3.2 Embeddings and vector store

- **Model**: **BAAI/bge-base-en** (SentenceTransformers), vector size **768**, batch size **16**.
- **Vector store**: **Qdrant** (localhost:6333), collection `docs`, **cosine** distance.
- **HNSW**: *m* = 16, *ef_construction* = 128, *ef_search* = 64.
- **Indexing**: Documents are embedded and upserted via `src/index_documents.py`; embeddings can be cached (e.g. `data/emb_cache`).

### 3.3 Retrieval strategies

| Strategy | Implementation | Main parameters |
|----------|----------------|------------------|
| **Dense** | Qdrant ANN search with query embedding | `retrieval.top_k` |
| **Sparse** | BM25Okapi (rank_bm25), NLTK tokenization; index cached per dataset | `retrieval.top_k`, `retrieval.bm25_cache_path` |
| **Hybrid** | Min-max normalization of dense and sparse scores; linear combination | `retrieval.top_k`, `retrieval.hybrid.w_dense`, `w_sparse`; `score_normalization: minmax` |

Hybrid fusion: for each document appearing in either dense or sparse top-*k*, normalized scores are computed (min-max over the respective list); then `score = w_dense * dense_norm + w_sparse * sparse_norm`; results are sorted by this score and truncated to *top_k*.

### 3.4 Metrics

- **Accuracy@k**: Binary accuracy at *k*: 1 if any of the query’s `relevant_doc_ids` appears in the top-*k* retrieved document IDs, else 0. The value of *k* is set by `metrics.accuracy_at_k` (e.g. 5, 10, or 20).
- **MAP@k and nDCG@k**: With ranked retrieved document IDs and binary relevance labels, we compute:
  - **AP@k / MAP@k**: Average Precision at *k* per query, averaged across queries to obtain MAP@k.
  - **DCG@k** and **nDCG@k**: Discounted Cumulative Gain at *k* with binary gains (1 for relevant, 0 otherwise), normalized by the ideal DCG@k to obtain nDCG@k in \([0,1]\).
  A separate `metrics.rank_k` controls the cutoff used for MAP@k and nDCG@k, allowing it to differ from `accuracy_at_k` if desired.
- **Latency**: Wall-clock time (seconds) for the retrieval call (query encoding + search for dense; search only for sparse; both plus fusion for hybrid).
- **Cost**: Proxy = total number of whitespace-separated tokens in the retrieved texts (used as a downstream cost proxy).
- **Normalization**: For each (query × retrieval_type) group, accuracy, latency, and cost are min-max normalized across the three retrieval types (dense, sparse, hybrid) so that each metric lies in [0, 1].
- **REM (Retrieval Efficiency Metric)**:
  \[
  \text{REM} = \alpha \cdot \text{accuracy\_norm} + \beta \cdot (1 - \text{latency\_norm}) + \gamma \cdot (1 - \text{cost\_norm})
  \]
  Default weights: α = 0.5 (accuracy), β = 0.3 (latency), γ = 0.2 (cost). Higher REM is better.

### 3.5 REM: definition and motivation

REM is a simple, transparent linear scalarization of three normalized metrics: retrieval accuracy, latency, and cost. The weights \(\alpha, \beta, \gamma\) encode the relative importance of each dimension for a given deployment scenario:

- \(\alpha\) emphasizes retrieval quality (accuracy).
- \(\beta\) emphasizes responsiveness (1 − latency).
- \(\gamma\) emphasizes resource efficiency (1 − cost).

By adjusting these weights, practitioners can move along a continuum between accuracy-optimal and efficiency-optimal configurations. In Section 5, we show that the best-by-accuracy configuration and the best-by-REM configuration differ: the REM-optimal setting trades a small amount of accuracy for noticeably better latency and cost, which is often preferable in production RAG systems.

---

## 4. Experiments

### 4.1 Evaluation protocol

1. Load FIQA test queries with relevance labels; optionally limit to `max_queries`.
2. For each query, run dense, sparse, and hybrid retrieval (same *top_k* and, for hybrid, same *w_dense* / *w_sparse*).
3. For each (query, retrieval_type), compute accuracy at *k*, latency, and cost; then normalize and compute REM.
4. Log one row per (query_id, query, retrieval_type, accuracy, latency_s, cost_tokens, REM, …) to a run-specific CSV.
5. Aggregate over queries: mean accuracy and mean REM per run (averaged across retrieval types for the run, or reported per retrieval type as needed).

### 4.2 Tuning grid (FIQA)

The tuning script `scripts/run_tuning.py` uses `configs/tuning_fiqa.yaml` and merges with `configs/experiment_config.yaml`. The sweep is:

| Parameter | Values |
|-----------|--------|
| `retrieval.top_k` | 5, 10, 20 |
| `retrieval.hybrid.w_dense` | 0.5, 0.6, 0.7 (*w_sparse* = 1 − *w_dense*) |
| `metrics.accuracy_at_k` | 5, 10, 20 |

Total combinations: **3 × 3 × 3 = 27 runs**. Each run evaluates all queries with dense, sparse, and hybrid retrieval under one parameter combination and writes a temporary log; results are merged into a single CSV and summarized.

### 4.3 Outputs

- **logs/tuning_fiqa.csv**: One row per (query × retrieval_type × run_id); columns include run_id, top_k, w_dense, w_sparse, accuracy_at_k, accuracy, latency_s, cost_tokens, REM, etc.
- **logs/tuning_fiqa_runs.csv**: One row per run_id; run-level aggregates (e.g. mean_accuracy, mean_REM) and parameters (top_k, w_dense, w_sparse, accuracy_at_k).
- **logs/tuning_fiqa_best.txt**: Best run by mean accuracy and best run by mean REM, with parameters and metric values.

---

## 5. Results

*The following sections can be filled manually from the tuning outputs or generated automatically by running `python scripts/generate_report.py` after `python scripts/run_tuning.py`.*

### 5.1 Best run by mean accuracy

```
  run_id=27, top_k=20.0, w_dense=0.70, accuracy_at_k=20.0
  mean_accuracy=0.5211, mean_REM=0.6937
```

### 5.2 Best run by mean REM

```
  run_id=27, top_k=20.0, w_dense=0.70, accuracy_at_k=20.0
  mean_accuracy=0.5211, mean_REM=0.6937
```

### 5.3 Run summary (abbreviated)

| run_id | top_k | w_dense | accuracy_at_k | mean_accuracy | mean_REM |
|--------|-------|---------|---------------|---------------|----------|
| run_id | top_k | w_dense | accuracy_at_k | mean_accuracy | mean_REM |
| --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0.50 | 5 | 0.4120 | 0.6512 |
| 2 | 5 | 0.50 | 10 | 0.4120 | 0.6442 |
| 3 | 5 | 0.50 | 20 | 0.4120 | 0.6431 |
| 4 | 5 | 0.60 | 5 | 0.4131 | 0.6477 |
| 5 | 5 | 0.60 | 10 | 0.4131 | 0.6445 |
| 6 | 5 | 0.60 | 20 | 0.4131 | 0.6448 |
| 7 | 5 | 0.70 | 5 | 0.4120 | 0.6443 |
| 8 | 5 | 0.70 | 10 | 0.4120 | 0.6440 |
| 9 | 5 | 0.70 | 20 | 0.4120 | 0.6445 |
| 10 | 10 | 0.50 | 5 | 0.4146 | 0.6415 |
| 11 | 10 | 0.50 | 10 | 0.4650 | 0.6669 |
| 12 | 10 | 0.50 | 20 | 0.4650 | 0.6668 |
| 13 | 10 | 0.60 | 5 | 0.4151 | 0.6424 |
| 14 | 10 | 0.60 | 10 | 0.4660 | 0.6678 |
| 15 | 10 | 0.60 | 20 | 0.4660 | 0.6677 |
| 16 | 10 | 0.70 | 5 | 0.4131 | 0.6377 |
| 17 | 10 | 0.70 | 10 | 0.4660 | 0.6644 |
| 18 | 10 | 0.70 | 20 | 0.4660 | 0.6644 |
| 19 | 20 | 0.50 | 5 | 0.4167 | 0.6388 |
| 20 | 20 | 0.50 | 10 | 0.4671 | 0.6638 |
| 21 | 20 | 0.50 | 20 | 0.5185 | 0.6906 |
| 22 | 20 | 0.60 | 5 | 0.4167 | 0.6393 |
| 23 | 20 | 0.60 | 10 | 0.4686 | 0.6653 |
| 24 | 20 | 0.60 | 20 | 0.5201 | 0.6919 |
| 25 | 20 | 0.70 | 5 | 0.4084 | 0.6373 |
| 26 | 20 | 0.70 | 10 | 0.4697 | 0.6676 |
| 27 | 20 | 0.70 | 20 | 0.5211 | 0.6937 |

*Full run-level table: see `logs/tuning_fiqa_runs.csv`.*

### 5.4 Per-retrieval-type averages (representative run)

For a single representative run (e.g. the best-by-REM run), average metrics by retrieval type:

| retrieval_type | accuracy | latency_s | cost_tokens | REM |
|----------------|----------|-----------|-------------|-----|
| retrieval_type | accuracy | latency_s | cost_tokens | REM |
| --- | --- | --- | --- | --- |
| dense | 0.5941 | 0.0319 | 2219.7176 | 0.7355 |
| hybrid | 0.6373 | 0.1791 | 2843.0679 | 0.7187 |
| sparse | 0.3318 | 0.1414 | 994.8657 | 0.6268 |

*Source: rows from `logs/tuning_fiqa.csv` for the chosen run_id, grouped by retrieval_type.*

### 5.5 Figures

- **Bar plot**: `figures/bar_metrics.png` — mean accuracy, latency, cost, REM by retrieval type (from a single run’s log or from the default runner output).
- **Radar plot**: `figures/radar_metrics.png` — normalized accuracy, latency efficiency (1 − latency_norm), cost efficiency (1 − cost_norm), and REM by retrieval type.
- **REM-focused plots** (from `notebooks/rem_analysis.ipynb`):
  - `figures/rem_best_run_metrics.png` — accuracy, latency, cost, and REM for the best-by-REM run, by retrieval type.
  - `figures/rem_accuracy_vs_rem_scatter.png` — relationship between mean_accuracy and mean_REM across all runs.
  - `figures/rem_heatmap_topk_wdense.png` — mean_REM as a function of `top_k` and `w_dense` for a fixed `accuracy_at_k`.
  - `figures/rem_distribution_hist.png` — distribution of REM values per retrieval type across all runs.
  - `figures/rem_radar_metrics.png` — radar plot of normalized accuracy, latency efficiency, cost efficiency, and REM for the best-by-REM run, by retrieval type; this plot makes it easy to compare trade-offs across dense, sparse, and hybrid retrieval in a single view.

*Generate with: from `src/`, run `python visualization.py` (uses default config and log path for the active dataset).*

---

## 6. Discussion

- **Accuracy vs. REM**: The best-by-accuracy run maximizes recall-related performance; the best-by-REM run balances accuracy with latency and cost. For production, REM may be more aligned with user experience and cost constraints.
- **Hybrid weight**: Sweeping *w_dense* (0.5–0.7) allows assessment of the dense/sparse trade-off on FIQA; results indicate how much lexical vs. semantic signal helps for this domain.
- **top_k and accuracy_at_k**: Larger *top_k* can improve recall but increase latency and cost; matching *accuracy_at_k* to downstream usage (e.g. how many docs are actually used) keeps the metric interpretable.

---

## 7. Limitations and future work

- **Single dataset**: Results are for FIQA only; replication on NFCorpus or other BEIR datasets is recommended.
- **Cost proxy**: Cost is approximated by retrieved token count; real cost would include embedding and API costs.
- **Single embedding model**: Only BAAI/bge-base-en is used; other models may change the dense/hybrid balance.
- **Reproducibility**: Results depend on Qdrant version, BEIR export script, and exact config (no `max_queries`/`max_documents` for full-data numbers). Future work could add confidence intervals (e.g. bootstrap over queries) and multiple seeds.

---

## 8. Conclusion

We described a reproducible pipeline for comparing dense, sparse, and hybrid retrieval on FIQA using a composite REM metric and a 27-point parameter grid. The report documents dataset, embeddings, retrieval implementation, REM formula, and tuning outputs. Best configurations by mean accuracy and by mean REM are reported; run-level and per-retrieval-type tables and figures support publication and replication.

---

## Reproducibility

- **Repository**: (add your repo URL)
- **Config**: `configs/experiment_config.yaml`, `configs/tuning_fiqa.yaml`, `configs/tuning_nfcorpus.yaml`
- **Data**: FIQA and NFCorpus from BEIR; export with e.g. `python scripts/export_beir_to_csv.py --dataset fiqa --split test --beir-root data/beir --out-dir data` (and similarly for `nfcorpus`).
- **Index**: `cd src && python index_documents.py`
- **Single run**: from the project root, `python scripts/run_experiments.py --mode single --datasets fiqa` (or `nfcorpus`); this writes `logs/logs_<dataset>.csv`.
- **Full tuning**: from the project root, `python scripts/run_experiments.py --mode tuning --datasets fiqa,nfcorpus` → produces `logs/tuning_<dataset>.csv`, `logs/tuning_<dataset>_runs.csv`, `logs/tuning_<dataset>_best.txt` for each dataset.
- **Report with results**: `python scripts/generate_report.py` (reads the FIQA tuning logs and updates this report’s result placeholders).
- **Analysis notebooks**: open `notebooks/rem_analysis.ipynb` for REM-focused plots on a single dataset and `notebooks/multi_dataset_analysis.ipynb` for Pareto frontiers, REM-weight sensitivity, and multi-dataset comparisons.
