from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import nltk
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
    def __init__(
        self,
        data_csv: str = "../data/business_corpus.csv",
        model_path: str = "./logs/bm25.pkl",
    ):
        self.data_csv = Path(data_csv)
        self.model_path = Path(model_path)
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

