"""Touchpoint 1 — Normalization.

Cleans and standardizes the raw ``input_record`` (make/model/variant/fuel/etc.)
using the synonym dictionary and an LLM, producing ``normalized_record``.

Public surface:
- ``normalize(state)``    — LangGraph node: reads MMVState, returns a partial
  state update ``{"normalized_record": ..., "llm_call_log": ...}``.
- ``normalize_record(input_record, service=None)`` — thin helper that returns
  just the normalized dict, for scripts/tests that don't build an MMVState.
"""

from __future__ import annotations

import json
from typing import Optional

from prompts.normalize_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from services.csv_loader import load_synonym_dictionary
from services.llm_service import LLMService
from state import MMVState

# Canonical shape of the normalized record; every key is always present.
_NORMALIZED_KEYS = (
    "raw_input",
    "make",
    "model",
    "variant",
    "fuel_type",
    "transmission",
    "cc",
    "seating_capacity",
    "year",
    "applied_synonyms",
)

# Reused across calls so the Gemini client / synonym dict load only once.
_service: Optional[LLMService] = None
_synonyms: Optional[dict] = None


def _get_service() -> LLMService:
    global _service
    if _service is None:
        _service = LLMService()
    return _service


def _get_synonyms() -> dict:
    global _synonyms
    if _synonyms is None:
        _synonyms = load_synonym_dictionary()
    return _synonyms


def _coerce_int(value: object) -> Optional[int]:
    """Best-effort coercion of cc / seating_capacity to int, else None."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):  # guard: bools are ints in Python
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _clean(record: dict, raw_input: str) -> dict:
    """Force the model output into the canonical normalized shape."""
    out = {k: record.get(k) for k in _NORMALIZED_KEYS}
    # Always trust our own raw_input over whatever the model echoed back.
    out["raw_input"] = raw_input
    out["cc"] = _coerce_int(out.get("cc"))
    out["seating_capacity"] = _coerce_int(out.get("seating_capacity"))
    if not isinstance(out.get("applied_synonyms"), list):
        out["applied_synonyms"] = []
    # Normalize empty strings to None for the string fields.
    for key in ("make", "model", "variant", "fuel_type", "transmission", "year"):
        val = out.get(key)
        if isinstance(val, str):
            val = val.strip()
            out[key] = val or None
    return out


def _extract_raw_input(input_record: dict) -> str:
    """Pull a display/free-text form of the raw input for auditing."""
    if "raw_input" in input_record and input_record["raw_input"]:
        return str(input_record["raw_input"])
    # Structured input: stitch the known fields into a readable string.
    parts = [
        str(input_record[k])
        for k in ("make", "model", "variant", "fuel_type", "transmission")
        if input_record.get(k)
    ]
    return " ".join(parts) if parts else json.dumps(input_record, ensure_ascii=False)


def normalize_record(
    input_record: dict, service: Optional[LLMService] = None
) -> dict:
    """Normalize a single raw input record into the canonical shape."""
    service = service or _get_service()
    synonyms = _get_synonyms()
    raw_input = _extract_raw_input(input_record)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        synonym_dictionary=json.dumps(synonyms, ensure_ascii=False, indent=2),
        input_record=json.dumps(input_record, ensure_ascii=False),
    )
    data = service.generate_json("normalization", SYSTEM_PROMPT, user_prompt)
    return _clean(data, raw_input)


def normalize(state: MMVState) -> dict:
    """Return a partial state update: {'normalized_record': ...}."""
    service = _get_service()
    input_record = state["input_record"]
    normalized = normalize_record(input_record, service=service)
    return {
        "normalized_record": normalized,
        "llm_call_log": list(service.call_log),
    }
