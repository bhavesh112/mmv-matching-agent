"""Touchpoint 2 — Query reformulation.

Expands the ``normalized_record`` into 2-4 ``search_queries`` (e.g. strict vs.
broad, plus a natural-language phrasing) to drive fuzzy and embedding retrieval.

Public surface:
- ``reformulate(state)`` — LangGraph node: reads MMVState, returns a partial
  state update ``{"search_queries": [...], "llm_call_log": ...}``.
- ``reformulate_queries(normalized_record, service=None)`` — helper returning
  just the list of query strings, for scripts/tests.
"""

from __future__ import annotations

import json
from typing import Optional

from prompts.reformulate_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from services.llm_service import LLMService
from state import MMVState

_MIN_QUERIES = 2
_MAX_QUERIES = 4

# Fields used to reconstruct a fallback query if the model returns nothing usable.
_STRICT_FIELDS = ("make", "model", "variant", "fuel_type", "transmission")

_service: Optional[LLMService] = None


def _get_service() -> LLMService:
    global _service
    if _service is None:
        _service = LLMService()
    return _service


def _fallback_queries(normalized_record: dict) -> list[str]:
    """Deterministically build queries from fields when the LLM output is junk.

    Produces a strict (all attributes) and a broad (make + model) query so the
    pipeline degrades gracefully rather than passing an empty query list on.
    """
    strict_parts = [
        str(normalized_record[f])
        for f in _STRICT_FIELDS
        if normalized_record.get(f)
    ]
    cc = normalized_record.get("cc")
    if cc:
        strict_parts.append(str(cc))

    broad_parts = [
        str(normalized_record[f])
        for f in ("make", "model")
        if normalized_record.get(f)
    ]

    queries: list[str] = []
    for parts in (strict_parts, broad_parts):
        q = " ".join(parts).strip()
        if q and q not in queries:
            queries.append(q)

    if not queries:
        # Last resort: fall back to the raw input echoed in the record.
        raw = str(normalized_record.get("raw_input") or "").strip()
        if raw:
            queries.append(raw)
    return queries


def _sanitize(queries: object, normalized_record: dict) -> list[str]:
    """Validate/clamp the model's queries to 2-4 non-empty, unique strings."""
    cleaned: list[str] = []
    if isinstance(queries, list):
        for q in queries:
            if not isinstance(q, str):
                continue
            q = q.strip()
            if q and q not in cleaned:
                cleaned.append(q)

    if len(cleaned) < _MIN_QUERIES:
        # Top up (or fully replace) with deterministic fallbacks.
        for q in _fallback_queries(normalized_record):
            if q not in cleaned:
                cleaned.append(q)

    return cleaned[:_MAX_QUERIES]


def reformulate_queries(
    normalized_record: dict, service: Optional[LLMService] = None
) -> list[str]:
    """Return 2-4 reformulated search query strings for a normalized record."""
    service = service or _get_service()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        normalized_record=json.dumps(normalized_record, ensure_ascii=False, indent=2),
    )
    data = service.generate_json("query_reformulation", SYSTEM_PROMPT, user_prompt)
    return _sanitize(data.get("queries"), normalized_record)


def reformulate(state: MMVState) -> dict:
    """Return a partial state update: {'search_queries': [...]}."""
    service = _get_service()
    queries = reformulate_queries(state["normalized_record"], service=service)
    return {
        "search_queries": queries,
        "llm_call_log": list(service.call_log),
    }
