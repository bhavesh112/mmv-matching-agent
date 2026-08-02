"""Embedding / semantic-search tool for the ReAct agent.

Embeds the query and each catalogue entry with a local sentence-transformers
model (all-MiniLM-L6-v2 — no external API) and ranks by cosine similarity.
Catalogue embeddings can be precomputed once and passed back in to avoid
re-encoding on every call (see RetrievalService).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from services.csv_loader import load_mmv_master
from tools.fuzzy_search_tool import build_candidate_text, row_to_record

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # lazily instantiated, module-level cache


def get_model():
    """Load (once) and return the shared SentenceTransformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "sentence-transformers is required for embedding_search. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_catalogue(
    df: Optional[pd.DataFrame] = None,
    model=None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Encode every catalogue row into an L2-normalized embedding matrix.

    Returns ``(df, embeddings)`` so the caller can cache both together.
    """
    if df is None:
        df = load_mmv_master()
    if model is None:
        model = get_model()

    texts = [build_candidate_text(row_to_record(row)) for _, row in df.iterrows()]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return df, np.asarray(embeddings, dtype=np.float32)


def embedding_search(
    query: str,
    df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    model=None,
    catalogue_embeddings: Optional[np.ndarray] = None,
) -> list[dict]:
    """Return the top_n semantically-similar MMV records for the query.

    Each result is the full record dict plus an ``embedding_score`` (cosine
    similarity in roughly [-1, 1], typically [0, 1]), sorted descending.
    """
    if df is None:
        df = load_mmv_master()
    if model is None:
        model = get_model()
    if catalogue_embeddings is None:
        _, catalogue_embeddings = embed_catalogue(df, model)

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    # Both sides are L2-normalized, so the dot product is cosine similarity.
    scores = catalogue_embeddings @ np.asarray(query_vec, dtype=np.float32)

    order = np.argsort(scores)[::-1][:top_n]
    results: list[dict] = []
    for pos in order:
        record = row_to_record(df.iloc[int(pos)])
        record["embedding_score"] = round(float(scores[int(pos)]), 4)
        results.append(record)
    return results
