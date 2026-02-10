from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dataset import load_documents
from embeddings import cache_embeddings
from vector_store_qdrant import QdrantConfig, QdrantVectorStore


def main(config_path: str = "../configs/experiment_config.yaml") -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    docs_csv = cfg["data"]["documents_csv"]
    cache_dir = Path(cfg["embeddings"].get("cache_dir", "data/emb_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "doc_embeddings.npy"

    docs = load_documents(docs_csv)
    texts = [d.text for d in docs]

    embeddings = cache_embeddings(texts, str(cache_path), config_path=config_path)
    assert embeddings.shape[1] == cfg["qdrant"]["vector_size"], (
        f"Embedding dimension {embeddings.shape[1]} does not match "
        f"configured vector_size {cfg['qdrant']['vector_size']}"
    )

    qcfg = QdrantConfig.from_yaml(config_path)
    store = QdrantVectorStore(qcfg)
    store.recreate_collection()

    payloads = []
    for doc, _ in zip(docs, embeddings):
        payloads.append(
            {
                "id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
            }
        )

    store.upsert_points(embeddings.tolist(), payloads)
    print(f"Indexed {len(docs)} documents into collection '{qcfg.collection_name}'.")


if __name__ == "__main__":
    main()

