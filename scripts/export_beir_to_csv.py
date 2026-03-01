from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def export_beir_dataset(
    beir_root: str,
    dataset_name: str,
    split: str,
    out_dir: str,
) -> None:
    """
    Convert a BEIR dataset into the CSV format expected by this project.

    Documents CSV columns:
        - id
        - title
        - text

    Queries CSV columns:
        - query_id
        - query
        - expected_answer
        - relevant_doc_ids (comma-separated list of document IDs)
    """
    beir_path = Path(beir_root) / dataset_name
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load BEIR-style files directly (no dependency on beir package).
    corpus_path = beir_path / "corpus.jsonl"
    queries_path = beir_path / "queries.jsonl"
    qrels_path = beir_path / "qrels" / f"{split}.tsv"

    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(
            f"Expected BEIR files not found under {beir_path}. "
            f"Looked for {corpus_path}, {queries_path}, {qrels_path}."
        )

    # corpus.jsonl: one JSON object per line with keys like _id, title, text
    corpus: Dict[str, Dict[str, str]] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("_id"))
            corpus[doc_id] = {
                "title": obj.get("title") or "",
                "text": obj.get("text") or "",
            }

    # queries.jsonl: one JSON object per line with keys like _id, text
    queries: Dict[str, str] = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id"))
            queries[qid] = obj.get("text") or ""

    # qrels/<split>.tsv: 3-col (query-id, corpus-id, score) or 4-col (query-id, unused, doc-id, relevance)
    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, doc_id, rel_str = parts
            elif len(parts) == 4:
                qid, _, doc_id, rel_str = parts
            else:
                continue
            if parts[0].lower() == "query-id":
                continue
            try:
                rel = int(rel_str)
            except ValueError:
                continue
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel

    # ---- Documents CSV ----
    docs_rows: List[Dict[str, str]] = []
    for doc_id, doc in corpus.items():
        title = (doc.get("title") or "").strip()
        text = (doc.get("text") or "").strip()

        if not title and text:
            title = text[:80]
        if not text:
            text = title or doc_id

        docs_rows.append(
            {
                "id": doc_id,
                "title": title,
                "text": text,
            }
        )

    docs_df = pd.DataFrame(docs_rows)
    docs_csv = out_path / f"{dataset_name}_{split}_documents.csv"
    docs_df.to_csv(docs_csv, index=False)

    # ---- Queries CSV ----
    queries_rows: List[Dict[str, str]] = []
    for qid, qtext in queries.items():
        qtext = (qtext or "").strip()

        # All relevant doc ids from qrels for this query
        rel_docs = list(qrels.get(qid, {}).keys())
        rel_str = ",".join(rel_docs)

        queries_rows.append(
            {
                "query_id": qid,
                "query": qtext,
                # For BEIR we rely on relevance labels; expected_answer is optional.
                "expected_answer": "",
                "relevant_doc_ids": rel_str,
            }
        )

    queries_df = pd.DataFrame(queries_rows)
    queries_csv = out_path / f"{dataset_name}_{split}_queries.csv"
    queries_df.to_csv(queries_csv, index=False)

    print(f"Wrote documents to: {docs_csv}")
    print(f"Wrote queries   to: {queries_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BEIR datasets to this project's CSV format.")
    parser.add_argument(
        "--beir-root",
        type=str,
        default="data/beir",
        help="Folder containing BEIR datasets (e.g. data/beir/fiqa/...).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fiqa",
        help="BEIR dataset name (e.g. fiqa, nfcorpus, scifact, ...).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to use (train/dev/test depending on dataset).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Directory where the CSV files will be written.",
    )
    args = parser.parse_args()

    export_beir_dataset(
        beir_root=args.beir_root,
        dataset_name=args.dataset,
        split=args.split,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

