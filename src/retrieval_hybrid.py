from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import yaml

from retrieval_dense import DenseRetriever, DenseResult
from retrieval_sparse import SparseRetriever, SparseResult


@dataclass
class HybridResult:
    doc_id: str
    title: str
    text: str
    score: float


def _minmax_norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    s_min, s_max = arr.min(), arr.max()
    if s_max == s_min:
        return [1.0 for _ in scores]
    return ((arr - s_min) / (s_max - s_min)).tolist()


class HybridRetriever:
    def __init__(self, config_path: str = "../configs/experiment_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.config_path = config_path
        self.top_k: int = int(cfg["retrieval"].get("top_k", 5))
        h_cfg = cfg["retrieval"]["hybrid"]
        self.w_dense: float = float(h_cfg.get("w_dense", 0.6))
        self.w_sparse: float = float(h_cfg.get("w_sparse", 0.4))
        self.dense = DenseRetriever(config_path=config_path)
        self.sparse = SparseRetriever(config_path=config_path)

    def search(self, query: str, top_k: int | None = None) -> List[HybridResult]:
        if top_k is None:
            top_k = self.top_k

        dense_results: List[DenseResult] = self.dense.search(query, top_k=top_k)
        sparse_results: List[SparseResult] = self.sparse.search(query, top_k=top_k)

        dense_scores = [r.score for r in dense_results]
        sparse_scores = [r.score for r in sparse_results]
        dense_norm = _minmax_norm(dense_scores)
        sparse_norm = _minmax_norm(sparse_scores)

        combined: Dict[str, Dict[str, float | str]] = {}

        for r, s in zip(dense_results, dense_norm):
            combined[r.doc_id] = {
                "title": r.title,
                "text": r.text,
                "dense": s,
                "sparse": 0.0,
            }
        for r, s in zip(sparse_results, sparse_norm):
            if r.doc_id not in combined:
                combined[r.doc_id] = {
                    "title": r.title,
                    "text": r.text,
                    "dense": 0.0,
                    "sparse": s,
                }
            else:
                combined[r.doc_id]["sparse"] = s

        results: List[HybridResult] = []
        for doc_id, info in combined.items():
            dense_s = float(info["dense"])
            sparse_s = float(info["sparse"])
            score = self.w_dense * dense_s + self.w_sparse * sparse_s
            results.append(
                HybridResult(
                    doc_id=doc_id,
                    title=str(info["title"]),
                    text=str(info["text"]),
                    score=score,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

