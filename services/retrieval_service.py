"""Retrieval service.

Owns the loaded MMV master and the precomputed catalogue embeddings, and runs
both the fuzzy and embedding search tools for a query. Merges/dedupes the two
result sets by ``mmv_id`` so every returned candidate carries *both* a
``fuzzy_score`` and an ``embedding_score``, plus a blended ``combined_score``
used for ranking.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from services.csv_loader import load_mmv_master
from tools.embedding_search_tool import embed_catalogue, embedding_search, get_model
from tools.fuzzy_search_tool import fuzzy_search

# Weights for blending the two normalized scores into combined_score.
_FUZZY_WEIGHT = 0.5
_EMBEDDING_WEIGHT = 0.5


class RetrievalService:
    """Serves combined fuzzy + embedding candidate lookups over the MMV master."""

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        self.df = df if df is not None else load_mmv_master()
        # Encode the catalogue once up front; reused for every query.
        self.model = get_model()
        _, self.catalogue_embeddings = embed_catalogue(self.df, self.model)

    def retrieve(self, query: str, top_n: int = 10) -> list[dict]:
        """Return the top_n combined candidates for ``query``.

        Both searches score the whole (small) catalogue, so the merge yields a
        complete pair of scores for each candidate rather than a lopsided union.
        """
        n = len(self.df)
        fuzzy_results = fuzzy_search(query, self.df, top_n=n)
        embedding_results = embedding_search(
            query,
            self.df,
            top_n=n,
            model=self.model,
            catalogue_embeddings=self.catalogue_embeddings,
        )

        merged: dict[str, dict] = {}

        for rec in fuzzy_results:
            mmv_id = rec["mmv_id"]
            entry = {k: v for k, v in rec.items() if k != "fuzzy_score"}
            entry["fuzzy_score"] = rec["fuzzy_score"]
            entry["embedding_score"] = None
            merged[mmv_id] = entry

        for rec in embedding_results:
            mmv_id = rec["mmv_id"]
            entry = merged.get(mmv_id)
            if entry is None:
                entry = {k: v for k, v in rec.items() if k != "embedding_score"}
                entry["fuzzy_score"] = None
                merged[mmv_id] = entry
            entry["embedding_score"] = rec["embedding_score"]

        candidates = list(merged.values())
        for entry in candidates:
            entry["combined_score"] = self._combined_score(
                entry.get("fuzzy_score"), entry.get("embedding_score")
            )

        candidates.sort(key=lambda e: e["combined_score"], reverse=True)
        return candidates[:top_n]

    @staticmethod
    def _combined_score(
        fuzzy_score: Optional[float], embedding_score: Optional[float]
    ) -> float:
        """Blend a 0-100 fuzzy score and a 0-1 cosine score onto one 0-1 scale."""
        fuzzy_norm = (fuzzy_score or 0.0) / 100.0
        embedding_norm = max(embedding_score or 0.0, 0.0)
        return round(
            _FUZZY_WEIGHT * fuzzy_norm + _EMBEDDING_WEIGHT * embedding_norm, 4
        )
