from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
import yaml


_MODEL: SentenceTransformer | None = None


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(config_path: str = "../configs/experiment_config.yaml") -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        cfg = _load_config(config_path)
        model_name = cfg["embeddings"]["model_name"]
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def embed_texts(
    texts: Sequence[str],
    config_path: str = "../configs/experiment_config.yaml",
    batch_size: int | None = None,
) -> np.ndarray:
    cfg = _load_config(config_path)
    if batch_size is None:
        batch_size = int(cfg["embeddings"].get("batch_size", 16))

    model = get_model(config_path)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings


def cache_embeddings(
    texts: Sequence[str],
    cache_path: str,
    config_path: str = "../configs/experiment_config.yaml",
) -> np.ndarray:
    """Compute and cache embeddings to a .npy file; load if already present."""
    cache_file = Path(cache_path)
    if cache_file.exists():
        return np.load(cache_file)

    emb = embed_texts(texts, config_path=config_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, emb)
    return emb

