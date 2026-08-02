"""Deterministic validation tool (NOT an LLM).

Applies hard business rules against a candidate match and returns a
validation_status. This is the gate the ReAct agent must pass before it is
allowed to emit an "approved"-eligible confidence: an LLM can be talked into a
plausible-sounding wrong answer, but these rules cannot.

Hard rules (any one failing => status "fail"):
  * fuel        — an explicitly stated input fuel must equal the candidate's.
  * transmission — an explicitly stated input transmission must equal the
                   candidate's. (Synonyms like dsl/AT are resolved upstream in
                   normalization, so a plain equality check is correct here.)
  * cc          — an explicitly stated input cc must equal the candidate's; cc
                  is the key discriminator between near-duplicate variants
                  (e.g. i20 Sportz 1197 vs 998) so no tolerance is allowed.
  * seating     — an explicitly stated input seating must equal the candidate's,
                  and the candidate's seating must be physically plausible.
  * year        — an explicitly stated input year must fall within the
                  candidate's [year_start, year_end] production range.

Only attributes actually present on the input are checked; a missing input
attribute is never itself a violation (that is ambiguity, handled elsewhere).
"""

from __future__ import annotations

from typing import Optional

# Physically plausible passenger-car seating range; anything outside is a data
# error on the candidate side.
_MIN_SEATS = 2
_MAX_SEATS = 10


def _norm_str(value: object) -> Optional[str]:
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


def _violation(rule: str, detail: str) -> dict:
    return {"rule": rule, "detail": detail}


def validate(normalized_record: dict, candidate: dict) -> dict:
    """Return ``{'validation_status', 'violations', 'checked'}`` for the candidate.

    ``validation_status`` is ``"pass"`` when no hard rule is violated, else
    ``"fail"``. ``violations`` lists each broken rule with a human-readable
    detail; ``checked`` lists the rules that were actually applicable (i.e. the
    input supplied that attribute) so callers can see how much was verified.
    """
    if not candidate:
        return {
            "validation_status": "fail",
            "violations": [_violation("candidate", "no candidate provided")],
            "checked": [],
        }

    violations: list[dict] = []
    checked: list[str] = []

    # --- fuel -------------------------------------------------------------
    in_fuel = _norm_str(normalized_record.get("fuel_type"))
    cand_fuel = _norm_str(candidate.get("fuel_type"))
    if in_fuel is not None:
        checked.append("fuel")
        if cand_fuel is not None and in_fuel != cand_fuel:
            violations.append(
                _violation(
                    "fuel",
                    f"input fuel '{normalized_record.get('fuel_type')}' != "
                    f"candidate '{candidate.get('fuel_type')}'",
                )
            )

    # --- transmission -----------------------------------------------------
    in_trans = _norm_str(normalized_record.get("transmission"))
    cand_trans = _norm_str(candidate.get("transmission"))
    if in_trans is not None:
        checked.append("transmission")
        if cand_trans is not None and in_trans != cand_trans:
            violations.append(
                _violation(
                    "transmission",
                    f"input transmission '{normalized_record.get('transmission')}' "
                    f"!= candidate '{candidate.get('transmission')}'",
                )
            )

    # --- cc ---------------------------------------------------------------
    in_cc = _norm_int(normalized_record.get("cc"))
    cand_cc = _norm_int(candidate.get("cc"))
    if in_cc is not None:
        checked.append("cc")
        if cand_cc is not None and in_cc != cand_cc:
            violations.append(
                _violation("cc", f"input cc {in_cc} != candidate cc {cand_cc}")
            )

    # --- seating ----------------------------------------------------------
    in_seats = _norm_int(normalized_record.get("seating_capacity"))
    cand_seats = _norm_int(candidate.get("seating_capacity"))
    if cand_seats is not None and not (_MIN_SEATS <= cand_seats <= _MAX_SEATS):
        violations.append(
            _violation(
                "seating",
                f"candidate seating {cand_seats} outside plausible "
                f"[{_MIN_SEATS}, {_MAX_SEATS}]",
            )
        )
    if in_seats is not None:
        checked.append("seating")
        if cand_seats is not None and in_seats != cand_seats:
            violations.append(
                _violation(
                    "seating",
                    f"input seating {in_seats} != candidate seating {cand_seats}",
                )
            )

    # --- year -------------------------------------------------------------
    in_year = _norm_int(normalized_record.get("year"))
    start = _norm_int(candidate.get("year_start"))
    end = _norm_int(candidate.get("year_end"))
    if in_year is not None:
        checked.append("year")
        if (start is not None and in_year < start) or (
            end is not None and in_year > end
        ):
            violations.append(
                _violation(
                    "year",
                    f"input year {in_year} outside candidate range "
                    f"[{start}, {end}]",
                )
            )

    return {
        "validation_status": "fail" if violations else "pass",
        "violations": violations,
        "checked": checked,
    }
