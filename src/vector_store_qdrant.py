from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import yaml
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchParams


@dataclass
class QdrantConfig:
    host: str
    port: int
    collection_name: str
    distance: str
    vector_size: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int

    @classmethod
    def from_yaml(cls, path: str) -> "QdrantConfig":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        q = cfg["qdrant"]
        return cls(
            host=q.get("host", "localhost"),
            port=int(q.get("port", 6333)),
            collection_name=q.get("collection_name", "business_docs"),
            distance=q.get("distance", "cosine"),
            vector_size=int(q.get("vector_size", 768)),
            hnsw_m=int(q.get("hnsw", {}).get("m", 16)),
            hnsw_ef_construction=int(q.get("hnsw", {}).get("ef_construction", 128)),
            hnsw_ef_search=int(q.get("hnsw", {}).get("ef_search", 64)),
        )


class QdrantVectorStore:
    """Thin wrapper around Qdrant for dense retrieval experiments."""

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client = QdrantClient(host=config.host, port=config.port)

    def recreate_collection(self) -> None:
        distance_enum = Distance.COSINE if self.config.distance.lower() == "cosine" else Distance.DOT

        self.client.recreate_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(size=self.config.vector_size, distance=distance_enum),
        )

        # Set HNSW params if supported
        self.client.update_collection(
            collection_name=self.config.collection_name,
            optimizer_config=None,
            hnsw_config={
                "m": self.config.hnsw_m,
                "ef_construct": self.config.hnsw_ef_construction,
            },
        )

    def upsert_points(
        self,
        vectors: Iterable[List[float]],
        payloads: Iterable[Dict[str, Any]],
        batch_size: int = 200,
    ) -> None:
        """Upsert points in batches to stay under Qdrant's HTTP payload size limit (~32 MB)."""
        vectors_list = list(vectors)
        payloads_list = list(payloads)
        if not vectors_list:
            return
        for start in range(0, len(vectors_list), batch_size):
            end = min(start + batch_size, len(vectors_list))
            batch_points: List[PointStruct] = []
            for i in range(start, end):
                # Qdrant point id must be int or UUID; keep original doc id in payload for retrieval
                batch_points.append(
                    PointStruct(
                        id=i,
                        vector=vectors_list[i],
                        payload=payloads_list[i],
                    )
                )
            self.client.upsert(collection_name=self.config.collection_name, points=batch_points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        with_payload: bool = True,
    ):
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=with_payload,
            search_params=SearchParams(hnsw_ef=self.config.hnsw_ef_search),
        )
        return response.points

