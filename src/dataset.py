from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    tags: Optional[str] = None


@dataclass
class QueryExample:
    query_id: str
    query: str
    expected_answer: str
    relevant_doc_ids: Optional[List[str]] = None


def load_documents(csv_path: str) -> List[Document]:
    """Load business documents from CSV.

    Expected columns: id, title, text, (optional) tags.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Documents CSV not found at {path}")

    df = pd.read_csv(path)
    required = {"id", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Documents CSV missing required columns: {missing}")

    docs: List[Document] = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                doc_id=str(row["id"]),
                title=str(row["title"]),
                text=str(row["text"]),
                tags=str(row["tags"]) if "tags" in df.columns and not pd.isna(row["tags"]) else None,
            )
        )
    return docs


def load_queries(csv_path: str) -> List[QueryExample]:
    """Load query–answer pairs.

    Expected columns: query_id, query, expected_answer, (optional) relevant_doc_ids.
    If relevant_doc_ids is present, it should be a comma-separated list of IDs.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries CSV not found at {path}")

    df = pd.read_csv(path)
    required = {"query_id", "query", "expected_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Queries CSV missing required columns: {missing}")

    queries: List[QueryExample] = []
    for _, row in df.iterrows():
        rel_ids: Optional[List[str]] = None
        if "relevant_doc_ids" in df.columns and isinstance(row.get("relevant_doc_ids"), str):
            rel_ids = [s.strip() for s in row["relevant_doc_ids"].split(",") if s.strip()]

        queries.append(
            QueryExample(
                query_id=str(row["query_id"]),
                query=str(row["query"]),
                expected_answer=str(row["expected_answer"]),
                relevant_doc_ids=rel_ids,
            )
        )
    return queries


def simple_chunk_documents(
    documents: Sequence[Document],
    max_chars: int = 800,
    overlap: int = 100,
) -> List[Tuple[Document, str]]:
    """Optionally chunk long documents into overlapping text spans.

    Returns (original_document, chunk_text) pairs.
    """
    chunks: List[Tuple[Document, str]] = []
    for doc in documents:
        text = doc.text
        if len(text) <= max_chars:
            chunks.append((doc, text))
            continue

        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            chunks.append((doc, chunk))
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
    return chunks

