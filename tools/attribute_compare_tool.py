"""Attribute-comparison tool for the ReAct agent.

Given a candidate MMV record and the normalized input, reports per-attribute
agreement (make/model/variant/fuel/transmission/cc/seating/year) so the agent
can discriminate near-duplicates (e.g. Swift VXI vs VXI+, i20 Sportz 1197 vs
998). This is a *pure, deterministic* helper — no LLM — that just surfaces the
diffs; the judgement of what those diffs mean is left to the agent.
"""

from __future__ import annotations

from typing import Optional

# Attributes compared as case-insensitive strings.
_STRING_FIELDS = ("make", "model", "variant", "fuel_type", "transmission")
# Attributes compared as integers.
_NUMERIC_FIELDS = ("cc", "seating_capacity")


def _norm_str(value: object) -> Optional[str]:
    """Lower-case, trimmed string form, or None when the value is absent."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _norm_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _compare_string(field: str, input_rec: dict, candidate: dict) -> dict:
    inp = _norm_str(input_rec.get(field))
    cand = _norm_str(candidate.get(field))
    if inp is None:
        status = "input_missing"
    elif cand is None:
        status = "candidate_missing"
    elif inp == cand:
        status = "match"
    else:
        status = "mismatch"
    return {
        "attribute": field,
        "input": input_rec.get(field),
        "candidate": candidate.get(field),
        "status": status,
    }


def _compare_numeric(field: str, input_rec: dict, candidate: dict) -> dict:
    inp = _norm_int(input_rec.get(field))
    cand = _norm_int(candidate.get(field))
    if inp is None:
        status = "input_missing"
    elif cand is None:
        status = "candidate_missing"
    elif inp == cand:
        status = "match"
    else:
        status = "mismatch"
    return {
        "attribute": field,
        "input": input_rec.get(field),
        "candidate": candidate.get(field),
        "status": status,
    }


def _compare_year(input_rec: dict, candidate: dict) -> dict:
    """Year is a point on the input side, a [start, end] range on the candidate."""
    inp = _norm_int(input_rec.get("year"))
    start = _norm_int(candidate.get("year_start"))
    end = _norm_int(candidate.get("year_end"))
    candidate_range = None
    if start is not None or end is not None:
        candidate_range = [start, end]
    if inp is None:
        status = "input_missing"
    elif start is None and end is None:
        status = "candidate_missing"
    elif (start is None or inp >= start) and (end is None or inp <= end):
        status = "match"
    else:
        status = "mismatch"
    return {
        "attribute": "year",
        "input": input_rec.get("year"),
        "candidate": candidate_range,
        "status": status,
    }


def attribute_compare(normalized_record: dict, candidate: dict) -> dict:
    """Return a per-attribute comparison report between input and candidate.

    Each attribute entry carries ``input``, ``candidate`` and a ``status`` of
    ``match`` / ``mismatch`` / ``input_missing`` / ``candidate_missing``. A
    ``summary`` block rolls up the counts and lists the mismatching / matching
    attributes so the agent can reason over the diff at a glance.
    """
    attributes = []
    for field in _STRING_FIELDS:
        attributes.append(_compare_string(field, normalized_record, candidate))
    for field in _NUMERIC_FIELDS:
        attributes.append(_compare_numeric(field, normalized_record, candidate))
    attributes.append(_compare_year(normalized_record, candidate))

    counts: dict[str, int] = {}
    for entry in attributes:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1

    return {
        "mmv_id": candidate.get("mmv_id"),
        "attributes": attributes,
        "summary": {
            "counts": counts,
            "matches": [a["attribute"] for a in attributes if a["status"] == "match"],
            "mismatches": [
                a["attribute"] for a in attributes if a["status"] == "mismatch"
            ],
            "input_missing": [
                a["attribute"] for a in attributes if a["status"] == "input_missing"
            ],
        },
    }
