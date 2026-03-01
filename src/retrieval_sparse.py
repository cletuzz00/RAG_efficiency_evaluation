from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import nltk
import yaml
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


_NLTK_DOWNLOADED = False


def _ensure_nltk():
    global _NLTK_DOWNLOADED
    if _NLTK_DOWNLOADED:
        return
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_DOWNLOADED = True


@dataclass
class SparseResult:
    doc_id: str
    title: str
    text: str
    score: float


class SparseRetriever:
    def __init__(self, config_path: str = "../configs/experiment_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg["data"]
        dataset = data_cfg.get("dataset")
        if dataset is not None:
            self.data_csv = Path(data_cfg.get("documents_csv") or f"../data/{dataset}_test_documents.csv")
            self.model_path = Path(cfg["retrieval"].get("bm25_cache_path") or f"../logs/bm25_{dataset}.pkl")
        else:
            self.data_csv = Path(data_cfg["documents_csv"])
            self.model_path = Path(cfg["retrieval"].get("bm25_cache_path", "logs/bm25.pkl"))
        self._max_documents = data_cfg.get("max_documents")
        if self._max_documents is not None:
            self.model_path = self.model_path.with_stem(
                self.model_path.stem + "_" + str(self._max_documents)
            )
        self._bm25: BM25Okapi | None = None
        self._df: pd.DataFrame | None = None
        self._load_or_train()

    def _load_or_train(self) -> None:
        _ensure_nltk()
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                obj = pickle.load(f)
            self._bm25 = obj["bm25"]
            self._df = obj["df"]
            return

        df = pd.read_csv(self.data_csv)
        if self._max_documents is not None:
            df = df.head(self._max_documents)
        corpus = [word_tokenize(str(text).lower()) for text in df["text"]]
        bm25 = BM25Okapi(corpus)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"bm25": bm25, "df": df}, f)
        self._bm25 = bm25
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        assert self._df is not None
        return self._df

    @property
    def bm25(self) -> BM25Okapi:
        assert self._bm25 is not None
        return self._bm25

    def search(self, query: str, top_k: int = 5) -> List[SparseResult]:
        _ensure_nltk()
        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:top_k]

        results: List[SparseResult] = []
        for i in idx:
            row = self.df.iloc[i]
            results.append(
                SparseResult(
                    doc_id=str(row["id"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                    score=float(scores[i]),
                )
            )
        return results

