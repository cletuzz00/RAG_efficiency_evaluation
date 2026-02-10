from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence

import yaml

from embeddings import embed_texts
from vector_store_qdrant import QdrantConfig, QdrantVectorStore


@dataclass
class DenseResult:
    doc_id: str
    title: str
    text: str
    score: float
    latency_s: float


class DenseRetriever:
    def __init__(self, config_path: str = "../configs/experiment_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.config_path = config_path
        self.top_k: int = int(cfg["retrieval"].get("top_k", 5))
        self.qdrant_cfg = QdrantConfig.from_yaml(config_path)
        self.store = QdrantVectorStore(self.qdrant_cfg)

    def search(self, query: str, top_k: int | None = None) -> List[DenseResult]:
        if top_k is None:
            top_k = self.top_k

        start = time.perf_counter()
        query_vec = embed_texts([query], config_path=self.config_path)[0].tolist()
        hits = self.store.search(query_vector=query_vec, top_k=top_k)
        latency = time.perf_counter() - start

        results: List[DenseResult] = []
        for h in hits:
            payload = h.payload or {}
            results.append(
                DenseResult(
                    doc_id=str(payload.get("id", h.id)),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                    score=float(h.score),
                    latency_s=latency,
                )
            )
        return results

