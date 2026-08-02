"""Fuzzy string-search tool for the ReAct agent.

Ranks MMV master rows by RapidFuzz ``token_sort_ratio`` against a free-text
query and returns the top-N candidates with scores. Token-sort makes the match
order-insensitive, so "Swift Maruti VXI" scores the same as "Maruti Swift VXI".
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

from services.csv_loader import load_mmv_master

# Fields concatenated into the searchable text for each catalogue row.
_TEXT_FIELDS = ["make", "model", "variant", "fuel_type", "transmission"]


def _clean_value(value: object) -> Optional[object]:
    """Convert pandas/NumPy NA and scalars into plain Python values."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value
    # Nullable Int64 / NumPy integers -> int
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def row_to_record(row: pd.Series) -> dict:
    """Turn a DataFrame row into a clean, NA-free dict."""
    return {col: _clean_value(row[col]) for col in row.index}


def build_candidate_text(record: dict) -> str:
    """Build the searchable string for a catalogue record.

    Includes make/model/variant/fuel/transmission plus CC and seating so
    queries that mention those (e.g. "1197", "7 seater") can still match.
    """
    parts = [str(record[f]) for f in _TEXT_FIELDS
             if record.get(f) not in (None, "") and not pd.isna(record.get(f))]
    cc = record.get("cc")
    if cc not in (None, "") and not pd.isna(cc):
        parts.append(f"{int(cc)}cc")
    seats = record.get("seating_capacity")
    if seats not in (None, "") and not pd.isna(seats):
        parts.append(f"{int(seats)} seater")
    return " ".join(parts)


def fuzzy_search(
    query: str,
    df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
) -> list[dict]:
    """Return the top_n fuzzy-matched MMV records for the query.

    Each result is the full record dict plus a ``fuzzy_score`` in [0, 100],
    sorted by score descending.
    """
    if df is None:
        df = load_mmv_master()

    # Map DataFrame index -> searchable text for process.extract.
    choices = {
        idx: build_candidate_text(row_to_record(row))
        for idx, row in df.iterrows()
    }

    matches = process.extract(
        query,
        choices,
        scorer=fuzz.token_sort_ratio,
        limit=top_n,
    )

    results: list[dict] = []
    for _text, score, idx in matches:
        record = row_to_record(df.loc[idx])
        record["fuzzy_score"] = round(float(score), 2)
        results.append(record)
    return results
