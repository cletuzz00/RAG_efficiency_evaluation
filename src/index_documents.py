from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dataset import load_documents
from embeddings import cache_embeddings
from vector_store_qdrant import QdrantConfig, QdrantVectorStore


def main(config_path: str = "../configs/experiment_config.yaml") -> None:
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to project root (parent of config dir) so this works from any cwd
    project_root = config_path.parent.parent
    data_cfg = cfg["data"]
    dataset = data_cfg.get("dataset")
    cache_dir = project_root / "data" / "emb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dataset is not None:
        docs_csv = data_cfg.get("documents_csv")
        if docs_csv is None:
            docs_csv = str(project_root / "data" / f"{dataset}_test_documents.csv")
        else:
            docs_csv = str((project_root / docs_csv.replace("../", "")).resolve() if docs_csv.startswith("../") else docs_csv)
        cache_path = cache_dir / f"doc_embeddings_{dataset}.npy"
    else:
        docs_csv = data_cfg["documents_csv"]
        if docs_csv.startswith("../"):
            docs_csv = str((project_root / docs_csv.replace("../", "")).resolve())
        cache_path = cache_dir / "doc_embeddings.npy"

    docs = load_documents(docs_csv)
    max_documents = cfg["data"].get("max_documents")
    if max_documents is not None:
        docs = docs[: int(max_documents)]
        print(f"Capped to {len(docs)} documents (max_documents={max_documents}).")
    texts = [d.text for d in docs]

    embeddings = cache_embeddings(texts, str(cache_path), config_path=str(config_path))
    assert embeddings.shape[1] == cfg["qdrant"]["vector_size"], (
        f"Embedding dimension {embeddings.shape[1]} does not match "
        f"configured vector_size {cfg['qdrant']['vector_size']}"
    )

    qcfg = QdrantConfig.from_yaml(str(config_path))
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

