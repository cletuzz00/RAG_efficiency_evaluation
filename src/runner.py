from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
import pandas as pd
import yaml

# Load .env from project root so HF_TOKEN is set for Hugging Face
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from dataset import load_queries
from metrics import (
    RemWeights,
    accuracy_binary_at_k,
    average_precision_at_k,
    compute_rem,
    jaccard_overlap,
    load_rem_weights,
    minmax_normalize,
    ndcg_at_k,
    recall_at_k,
)
from retrieval_dense import DenseRetriever
from retrieval_sparse import SparseRetriever
from retrieval_hybrid import HybridRetriever


@dataclass
class RetrievalRecord:
    query_id: str
    query: str
    retrieval_type: str
    accuracy: float
    map_at_k: float
    ndcg_at_k: float
    recall_at_k: float
    latency_s: float
    cost_tokens: float


def _estimate_cost_tokens(texts: List[str]) -> float:
    # Simple proxy: number of whitespace-separated tokens across retrieved texts
    return float(sum(len(t.split()) for t in texts))


def run_experiments(config_path: str = "../configs/experiment_config.yaml") -> pd.DataFrame:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["data"].get("dataset")
    if dataset is not None:
        queries_csv = cfg["data"].get("queries_csv") or f"../data/{dataset}_test_queries.csv"
        logs_csv = cfg["logging"].get("logs_csv") or f"../logs/logs_{dataset}.csv"
    else:
        queries_csv = cfg["data"]["queries_csv"]
        logs_csv = cfg["logging"]["logs_csv"]

    accuracy_at_k = int(cfg["metrics"].get("accuracy_at_k", 5))
    rank_k = int(cfg["metrics"].get("rank_k", accuracy_at_k))

    retrieval_top_k = int(cfg["retrieval"].get("top_k", accuracy_at_k))

    queries = load_queries(queries_csv)
    queries = [q for q in queries if q.relevant_doc_ids]
    max_queries = cfg["data"].get("max_queries")
    if max_queries is not None:
        queries = queries[: int(max_queries)]
    print(f"Running on {len(queries)} labeled queries" + (f" (max_queries={max_queries})." if max_queries is not None else "."))

    dense = DenseRetriever(config_path=config_path)
    sparse = SparseRetriever(config_path=config_path)
    hybrid = HybridRetriever(config_path=config_path)

    records: List[RetrievalRecord] = []

    for q in queries:
        # Dense
        t0 = time.perf_counter()
        dense_results = dense.search(q.query, top_k=retrieval_top_k)
        latency_dense = time.perf_counter() - t0
        dense_ids = [r.doc_id for r in dense_results]
        dense_texts = [r.text for r in dense_results]
        acc_dense = accuracy_binary_at_k(dense_ids, q, k=accuracy_at_k)
        map_dense = average_precision_at_k(dense_ids, q, k=rank_k)
        ndcg_dense = ndcg_at_k(dense_ids, q, k=rank_k)
        rec_dense = recall_at_k(dense_ids, q, k=accuracy_at_k)
        # fall back to Jaccard on text if no relevance labels
        if not q.relevant_doc_ids:
            joint_text = " ".join(dense_texts)
            acc_dense = jaccard_overlap(joint_text, q.expected_answer)
        cost_dense = _estimate_cost_tokens(dense_texts)
        records.append(
            RetrievalRecord(
                query_id=q.query_id,
                query=q.query,
                retrieval_type="dense",
                accuracy=acc_dense,
                map_at_k=map_dense,
                ndcg_at_k=ndcg_dense,
                recall_at_k=rec_dense,
                latency_s=latency_dense,
                cost_tokens=cost_dense,
            )
        )

        # Sparse
        t0 = time.perf_counter()
        sparse_results = sparse.search(q.query, top_k=retrieval_top_k)
        latency_sparse = time.perf_counter() - t0
        sparse_ids = [r.doc_id for r in sparse_results]
        sparse_texts = [r.text for r in sparse_results]
        acc_sparse = accuracy_binary_at_k(sparse_ids, q, k=accuracy_at_k)
        map_sparse = average_precision_at_k(sparse_ids, q, k=rank_k)
        ndcg_sparse = ndcg_at_k(sparse_ids, q, k=rank_k)
        rec_sparse = recall_at_k(sparse_ids, q, k=accuracy_at_k)
        if not q.relevant_doc_ids:
            joint_text = " ".join(sparse_texts)
            acc_sparse = jaccard_overlap(joint_text, q.expected_answer)
        cost_sparse = _estimate_cost_tokens(sparse_texts)
        records.append(
            RetrievalRecord(
                query_id=q.query_id,
                query=q.query,
                retrieval_type="sparse",
                accuracy=acc_sparse,
                map_at_k=map_sparse,
                ndcg_at_k=ndcg_sparse,
                recall_at_k=rec_sparse,
                latency_s=latency_sparse,
                cost_tokens=cost_sparse,
            )
        )

        # Hybrid
        t0 = time.perf_counter()
        hybrid_results = hybrid.search(q.query, top_k=retrieval_top_k)
        latency_hybrid = time.perf_counter() - t0
        hybrid_ids = [r.doc_id for r in hybrid_results]
        hybrid_texts = [r.text for r in hybrid_results]
        acc_hybrid = accuracy_binary_at_k(hybrid_ids, q, k=accuracy_at_k)
        map_hybrid = average_precision_at_k(hybrid_ids, q, k=rank_k)
        ndcg_hybrid = ndcg_at_k(hybrid_ids, q, k=rank_k)
        rec_hybrid = recall_at_k(hybrid_ids, q, k=accuracy_at_k)
        if not q.relevant_doc_ids:
            joint_text = " ".join(hybrid_texts)
            acc_hybrid = jaccard_overlap(joint_text, q.expected_answer)
        cost_hybrid = _estimate_cost_tokens(hybrid_texts)
        records.append(
            RetrievalRecord(
                query_id=q.query_id,
                query=q.query,
                retrieval_type="hybrid",
                accuracy=acc_hybrid,
                map_at_k=map_hybrid,
                ndcg_at_k=ndcg_hybrid,
                recall_at_k=rec_hybrid,
                latency_s=latency_hybrid,
                cost_tokens=cost_hybrid,
            )
        )

    # Build DataFrame
    df = pd.DataFrame([r.__dict__ for r in records])

    # Normalize metrics and compute REM (per-query normalization)
    def _normalize_per_query(df_in: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for _, g in df_in.groupby("query_id", sort=False):
            g = g.copy()
            g["accuracy_norm"] = minmax_normalize(g["accuracy"].tolist())
            g["latency_norm"] = minmax_normalize(g["latency_s"].tolist())
            g["cost_norm"] = minmax_normalize(g["cost_tokens"].tolist())
            parts.append(g)
        return pd.concat(parts, ignore_index=True)

    df = _normalize_per_query(df)

    weights: RemWeights = load_rem_weights(config_path)
    df["REM"] = df.apply(
        lambda r: compute_rem(
            accuracy_norm=float(r["accuracy_norm"]),
            latency_norm=float(r["latency_norm"]),
            cost_norm=float(r["cost_norm"]),
            weights=weights,
        ),
        axis=1,
    )

    logs_path = Path(logs_csv)
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(logs_path, index=False)

    print("Per-retrieval-type averages:")
    print(df.groupby("retrieval_type")[["accuracy", "latency_s", "cost_tokens", "REM"]].mean())

    return df


if __name__ == "__main__":
    run_experiments()

