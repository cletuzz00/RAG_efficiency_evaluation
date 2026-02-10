from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import yaml

from dataset import QueryExample


def minmax_normalize(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax == vmin:
        return [1.0 for _ in arr]
    return ((arr - vmin) / (vmax - vmin)).tolist()


def jaccard_overlap(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union


def accuracy_binary_at_k(
    retrieved_doc_ids: Sequence[str],
    query_example: QueryExample,
    k: int,
) -> float:
    """Return 1.0 if any relevant_doc_ids appear in top-k retrieved_doc_ids."""
    if not query_example.relevant_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    rel = set(query_example.relevant_doc_ids)
    return 1.0 if top_k & rel else 0.0


@dataclass
class RemWeights:
    alpha: float
    beta: float
    gamma: float


def load_rem_weights(config_path: str = "../configs/experiment_config.yaml") -> RemWeights:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    w = cfg["metrics"]["rem_weights"]
    return RemWeights(
        alpha=float(w.get("alpha", 0.5)),
        beta=float(w.get("beta", 0.3)),
        gamma=float(w.get("gamma", 0.2)),
    )


def compute_rem(
    accuracy_norm: float,
    latency_norm: float,
    cost_norm: float,
    weights: RemWeights,
) -> float:
    """REM = α * Acc + β * (1 - Lat) + γ * (1 - Cost)."""
    return (
        weights.alpha * accuracy_norm
        + weights.beta * (1.0 - latency_norm)
        + weights.gamma * (1.0 - cost_norm)
    )

