"""Reusable batch runner + trace builder for the HTTP layer.

The CLI driver (``app.py``) prints results to a terminal; the FastAPI layer
(``server.py``) needs the *same* pipeline but shaped into a JSON trace a UI can
render. This module is that shared middle: it runs one input record end to end
and flattens the resulting ``MMVState`` into a stable ``record result`` document
describing which of the six touchpoints fired, in order, with each one's output.

Two execution modes:

* ``"live"`` — the real pipeline: normalize -> reformulate -> retrieve ->
  ``graph.build_graph().invoke`` (ReAct reasoning -> confidence routing ->
  adversarial verifier -> explanation -> audit). Needs a Gemini API key.

* ``"mock"`` — a deterministic, no-LLM stand-in that still exercises the *real*
  deterministic tools (fuzzy retrieval, ``validate``, ``attribute_compare``) and
  synthesizes plausible reasoning/verifier/explanation outputs from them. Lets
  the UI be driven and demoed without spending Gemini quota; every touchpoint
  output it produces is flagged ``"mock": true`` so nothing is passed off as a
  real model call.

Both modes return the identical record-result shape, so the UI never has to know
which one produced it.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import pandas as pd

import config
from services.csv_loader import load_synonym_dictionary
from tools.attribute_compare_tool import attribute_compare
from tools.fuzzy_search_tool import fuzzy_search, row_to_record
from tools.validate_tool import validate

# The six touchpoints, in pipeline order. ``key`` matches the ``touchpoint``
# label the LLMService stamps on each call in ``llm_call_log``.
TOUCHPOINTS = (
    (1, "normalization", "Normalization"),
    (2, "query_reformulation", "Query reformulation"),
    (3, "reasoning", "ReAct reasoning"),
    (4, "verifier", "Adversarial verifier"),
    (5, "explanation", "Explanation"),
    (6, "feedback", "Feedback mining"),
)

_MATCH_FIELDS = (
    "mmv_id",
    "make",
    "model",
    "variant",
    "fuel_type",
    "transmission",
    "cc",
    "seating_capacity",
    "year_start",
    "year_end",
)


# ---------------------------------------------------------------------------
# Confidence tiers (mirror graph._route_by_confidence)
# ---------------------------------------------------------------------------


def confidence_tier(confidence: float) -> str:
    """Bucket a confidence into the graph's three routing tiers."""
    conf = float(confidence or 0.0)
    if conf > config.AUTO_APPROVE_CONFIDENCE_THRESHOLD:
        return "auto"      # -> adversarial verifier, then approve/review
    if conf >= config.REVIEW_CONFIDENCE_FLOOR:
        return "review"    # -> manual review
    return "reject"        # -> rejected


def thresholds() -> dict:
    return {
        "auto_approve": config.AUTO_APPROVE_CONFIDENCE_THRESHOLD,
        "review_floor": config.REVIEW_CONFIDENCE_FLOOR,
    }


# ---------------------------------------------------------------------------
# Trace builder: MMVState -> record result document
# ---------------------------------------------------------------------------


def _slim_match(record: Optional[dict]) -> Optional[dict]:
    if not record:
        return None
    return {k: record.get(k) for k in _MATCH_FIELDS if k in record}


def _calls_for(llm_call_log: list, key: str) -> list:
    return [c for c in (llm_call_log or []) if c.get("touchpoint") == key]


def _touchpoint_states(state: dict, mock: bool) -> list:
    """Assemble the six-station touchpoint trace from a finished state."""
    log = state.get("llm_call_log") or []
    normalized = state.get("normalized_record") or {}
    queries = state.get("search_queries") or []
    decision = state.get("reasoning_decision") or {}
    verifier = state.get("verifier_result")
    trace = state.get("reasoning_trace") or []
    error = state.get("error")

    # A normalized record that carries only raw_input never really normalized.
    normalized_fired = any(
        normalized.get(k) for k in ("make", "model", "variant", "fuel_type",
                                    "transmission", "cc", "applied_synonyms")
    ) or bool(_calls_for(log, "normalization"))

    stations = []
    for tid, key, name in TOUCHPOINTS:
        calls = _calls_for(log, key)
        entry = {"id": tid, "key": key, "name": name, "calls": calls,
                 "mock": mock, "status": "skipped", "summary": "", "output": None}

        if key == "normalization":
            if normalized_fired:
                syn = normalized.get("applied_synonyms") or []
                entry["status"] = "fired"
                entry["summary"] = (
                    f"{len(syn)} synonym(s) applied"
                    if syn else "Cleaned & standardized the raw input"
                )
                entry["output"] = {"normalized_record": normalized}
            else:
                entry["status"] = "error" if error else "skipped"
                entry["summary"] = "Did not run" if not error else "Failed before normalization"

        elif key == "query_reformulation":
            if queries:
                entry["status"] = "fired"
                entry["summary"] = f"{len(queries)} search query(s)"
                entry["output"] = {"queries": queries}
            else:
                entry["status"] = "error" if error else "skipped"
                entry["summary"] = "Did not run"

        elif key == "reasoning":
            if trace or decision:
                entry["status"] = "fired"
                entry["summary"] = (
                    f"{state.get('step_count', len(trace))} step(s) · "
                    f"validation {state.get('validation_status', 'n/a')}"
                )
                entry["output"] = {
                    "decision": {
                        "match": _slim_match(decision.get("match")),
                        "confidence": decision.get("confidence"),
                        "reason": decision.get("reason"),
                    },
                    "validation_status": state.get("validation_status"),
                    "step_count": state.get("step_count"),
                    "trace": trace,
                }
            else:
                entry["status"] = "error" if error else "skipped"
                entry["summary"] = "Did not run"

        elif key == "verifier":
            if verifier:
                passed = verifier.get("passed")
                entry["status"] = "fired"
                entry["summary"] = (
                    "Match survived scrutiny" if passed
                    else "Raised a substantive concern"
                )
                entry["output"] = verifier
            else:
                entry["status"] = "skipped"
                entry["summary"] = (
                    "Skipped — only the top confidence tier (> "
                    f"{config.AUTO_APPROVE_CONFIDENCE_THRESHOLD:g}) is verified"
                )

        elif key == "explanation":
            explanation = state.get("explanation")
            if explanation:
                entry["status"] = "fired"
                entry["summary"] = "Reviewer-readable summary produced"
                entry["output"] = {"explanation": explanation}
            else:
                entry["status"] = "skipped"
                entry["summary"] = "Did not run"

        elif key == "feedback":
            entry["status"] = "offline"
            entry["summary"] = "Batch/offline touchpoint — mines audit logs after a run"
            entry["output"] = None

        stations.append(entry)
    return stations


def build_record_result(input_record: dict, state: dict, mode: str) -> dict:
    """Flatten a finished ``MMVState`` into the UI-facing record result."""
    mock = mode == "mock"
    decision = state.get("reasoning_decision") or {}
    match = _slim_match(state.get("selected_match") or None)
    confidence = float(state.get("confidence") or 0.0)
    verifier = state.get("verifier_result")

    return {
        "record_id": (input_record or {}).get("input_id"),
        "raw_input": (input_record or {}).get("raw_input"),
        "input_meta": {
            k: v for k, v in (input_record or {}).items()
            if k not in ("input_id", "raw_input")
        },
        "status": state.get("final_decision", "review"),
        "confidence": round(confidence, 4),
        "confidence_tier": confidence_tier(confidence),
        "match": match,
        "validation_status": state.get("validation_status"),
        "verifier": {
            "fired": verifier is not None,
            **(verifier or {}),
        },
        "explanation": state.get("explanation", ""),
        "reasoning": {
            "reason": decision.get("reason", ""),
            "step_count": state.get("step_count", 0),
            "trace": state.get("reasoning_trace") or [],
        },
        "candidates": state.get("candidate_records") or [],
        "touchpoints": _touchpoint_states(state, mock),
        "llm_calls": state.get("llm_call_log") or [],
        "error": state.get("error"),
        "mock": mock,
        "override": None,
    }


# ---------------------------------------------------------------------------
# Mock (no-LLM) pipeline — deterministic, exercises the real tools
# ---------------------------------------------------------------------------

_FUEL_WORDS = {"petrol", "diesel", "cng", "electric", "hybrid"}
_TRANS_WORDS = {"manual", "automatic", "amt", "cvt", "ivt", "dct"}
# Discriminating attributes the input can omit, creating genuine ambiguity.
_DISCRIMINATORS = ("fuel_type", "transmission", "cc", "seating_capacity")
# Below this blended surface-similarity, a validation "pass" is not enough to
# trust a match — the input simply didn't state enough to check an off-catalogue
# make/model. Used only by the mock pipeline's reject gate.
_MATCH_SIMILARITY_FLOOR = 0.48


class MockPipeline:
    """Deterministic matcher over an MMV master — no network, no LLM.

    Runs the real fuzzy-search / validate / attribute_compare tools and
    synthesizes the three LLM touchpoints (normalization, reasoning, verifier)
    plus a templated explanation, so the UI has a complete, honestly-labelled
    trace to render without a Gemini key.
    """

    def __init__(self, master_df: pd.DataFrame) -> None:
        self.df = master_df
        self.records = [row_to_record(row) for _, row in master_df.iterrows()]
        try:
            self.synonyms = load_synonym_dictionary()
        except Exception:  # noqa: BLE001 - synonyms are a nicety, not required
            self.synonyms = {}
        self._makes = self._vocab("make")
        self._models = self._vocab("model")
        self._variants = self._vocab("variant")

    def _vocab(self, field: str) -> list:
        seen: list = []
        for rec in self.records:
            val = rec.get(field)
            if val and val not in seen:
                seen.append(str(val))
        # Longest first so multiword models ("Innova Crysta") win over substrings.
        return sorted(seen, key=len, reverse=True)

    # --- Touchpoint 1: normalization (deterministic) -------------------
    def normalize(self, raw_input: str) -> dict:
        text = raw_input or ""
        applied: list = []

        # Apply synonyms (longest source phrase first) to canonicalize tokens.
        for category, mapping in (self.synonyms or {}).items():
            for src, canon in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
                if re.search(rf"(?<!\w){re.escape(src)}(?!\w)", text, re.IGNORECASE):
                    text = re.sub(
                        rf"(?<!\w){re.escape(src)}(?!\w)", canon, text, flags=re.IGNORECASE
                    )
                    # Skip pure case-only rewrites (e.g. "Xuv700" -> "XUV700").
                    if src.lower() != canon.lower():
                        applied.append({"category": category.replace("_synonyms", ""),
                                        "from": src, "to": canon})

        low = text.lower()
        make = next((m for m in self._makes if m.lower() in low), None)
        model = next((m for m in self._models if m.lower() in low), None)
        variant = next(
            (v for v in self._variants
             if re.search(rf"(?<!\w){re.escape(v.lower())}(?!\w)", low)),
            None,
        )
        fuel = next((w for w in _FUEL_WORDS if re.search(rf"(?<!\w){w}(?!\w)", low)), None)
        trans = next((w for w in _TRANS_WORDS if re.search(rf"(?<!\w){w}(?!\w)", low)), None)
        # cc: an explicit "1197cc" (3-4 digits), else a standalone 4-digit engine
        # size. Deliberately NOT 3-digit bare numbers, so model tokens like the
        # "700" in "XUV 700" or "800" in "Maruti 800" are not mistaken for cc.
        cc = None
        cc_explicit = re.search(r"(?<!\d)(\d{3,4})\s*cc(?!\w)", low)
        if cc_explicit:
            cc = int(cc_explicit.group(1))
        else:
            cc_bare = re.search(r"(?<!\w)(\d{4})(?!\w)", low)
            if cc_bare and 1000 <= int(cc_bare.group(1)) <= 3500:
                cc = int(cc_bare.group(1))
        seat_match = re.search(r"(\d+)\s*seat", low)
        seats = int(seat_match.group(1)) if seat_match else None

        return {
            "raw_input": raw_input,
            "make": make,
            "model": model,
            "variant": variant,
            "fuel_type": fuel.capitalize() if fuel else None,
            "transmission": _canon_trans(trans),
            "cc": cc,
            "seating_capacity": seats,
            "year": None,
            "applied_synonyms": applied,
        }

    # --- Touchpoint 2: query reformulation -----------------------------
    def reformulate(self, normalized: dict) -> list:
        strict = " ".join(
            str(normalized[f]) for f in
            ("make", "model", "variant", "fuel_type", "transmission")
            if normalized.get(f)
        ).strip()
        broad = " ".join(
            str(normalized[f]) for f in ("make", "model") if normalized.get(f)
        ).strip()
        raw = str(normalized.get("raw_input") or "").strip()
        queries: list = []
        for q in (strict, broad, raw):
            if q and q not in queries:
                queries.append(q)
        return queries or [raw]

    def retrieve(self, queries: list, top_n: int = 8) -> list:
        merged: dict = {}
        for q in queries:
            for cand in fuzzy_search(q, self.df, top_n=len(self.df)):
                mid = cand.get("mmv_id")
                score = cand.get("fuzzy_score", 0.0)
                if mid not in merged or score > merged[mid].get("fuzzy_score", 0.0):
                    entry = dict(cand)
                    entry["embedding_score"] = None
                    entry["combined_score"] = round(score / 100.0, 4)
                    merged[mid] = entry
        ranked = sorted(merged.values(), key=lambda c: c["combined_score"], reverse=True)
        return ranked[:top_n]

    # --- Touchpoint 3: reasoning (synthesized from real tool outputs) --
    def run(self, input_record: dict) -> dict:
        raw = input_record.get("raw_input", "")
        normalized = self.normalize(raw)
        queries = self.reformulate(normalized)
        candidates = self.retrieve(queries)

        trace: list = []
        validations: dict = {}

        if not candidates:
            return self._assemble(input_record, normalized, queries, candidates,
                                  None, 0.15, trace, "No candidates retrieved.",
                                  validations, "not_validated")

        # Explore candidates top-down (like the ReAct agent): compare + validate
        # each until one clears both — a real match must satisfy every stated
        # attribute AND pass the hard rules. Cap the exploration so a reject still
        # produces a short, readable trace.
        step = 0
        leading = None
        eligibles: list = []
        for cand in candidates[:5]:
            cmp = attribute_compare(normalized, cand)
            step += 1
            trace.append(_step(step, f"Diff candidate {cand['mmv_id']} against the "
                                     f"normalized input, attribute by attribute.",
                               "attribute_compare", {"mmv_id": cand["mmv_id"]}, cmp))
            v = validate(normalized, cand)
            validations[cand["mmv_id"]] = v
            step += 1
            trace.append(_step(step, f"Gate {cand['mmv_id']} through the hard business "
                                     f"rules (fuel / transmission / cc / seating / year).",
                               "validate", {"mmv_id": cand["mmv_id"]}, v))
            eligible = (v["validation_status"] == "pass"
                        and not cmp["summary"]["mismatches"])
            if eligible:
                eligibles.append(cand)
                if leading is None:
                    leading = cand
                # One rival is enough to establish ambiguity; stop exploring.
                if len(eligibles) >= 2:
                    break
            if leading is not None and len(eligibles) >= 2:
                break

        # A candidate can pass validation on the few stated attributes yet still be
        # a poor match overall (an out-of-catalogue make/model the input never let
        # us check). Require real surface similarity before trusting the leader.
        rejected_low_sim = False
        if leading is not None and leading["combined_score"] < _MATCH_SIMILARITY_FLOOR:
            step += 1
            trace.append(_step(step, f"{leading['mmv_id']} clears the stated attributes "
                                     f"but surface similarity is only "
                                     f"{leading['combined_score']:.2f} — too low to trust "
                                     f"as a genuine catalogue match.",
                               "finish", {"match": None, "confidence": 0.4},
                               "rejected on low similarity"))
            leading, eligibles, rejected_low_sim = None, [], True

        # Decide + score into the graph's three routing tiers.
        if leading is None:
            match, conf = None, 0.35
            reason = ("No catalogue row both satisfies every stated attribute and is "
                      "sufficiently similar; rejecting rather than forcing a match.")
            validation_status = "reject"
        else:
            rival = next((c for c in eligibles if c["mmv_id"] != leading["mmv_id"]), None)
            twin_fields = _differing_fields(leading, rival) if rival else []
            ambiguous = bool(
                rival
                and abs(leading["combined_score"] - rival["combined_score"]) <= 0.08
                and twin_fields
                and all(f in _DISCRIMINATORS and normalized.get(f) in (None, "")
                        for f in twin_fields)
            )
            if ambiguous:
                step += 1
                trace.append(_step(
                    step, f"{leading['mmv_id']} and {rival['mmv_id']} both satisfy every "
                          f"stated attribute and differ only on {', '.join(twin_fields)} "
                          f"— which the input never specified. Ambiguous; escalate to "
                          f"human review.",
                    "finish", {"match": leading["mmv_id"], "confidence": 0.88},
                    "escalated to review (ambiguous twin candidate)"))
                match, conf = leading, 0.88
                reason = (f"{leading['mmv_id']} and {rival['mmv_id']} are both valid; they "
                          f"differ only on {', '.join(twin_fields)}, absent from the input "
                          f"— routing to human review.")
                validation_status = validations[leading["mmv_id"]]["validation_status"]
            else:
                match, conf = leading, 0.96
                reason = (f"{leading['mmv_id']} uniquely satisfies every stated attribute "
                          f"and cleared hard-rule validation with no equally-valid rival.")
                validation_status = validations[leading["mmv_id"]]["validation_status"]
                step += 1
                trace.append(_step(step, "One candidate uniquely clears every check. "
                                         "Commit to it.",
                                   "finish", {"match": leading["mmv_id"], "confidence": conf},
                                   reason))

        if leading is None and not rejected_low_sim:
            step += 1
            trace.append(_step(step, "Every examined candidate fails a hard rule or "
                                     "contradicts a stated attribute. Reject.",
                               "finish", {"match": None, "confidence": conf}, reason))

        return self._assemble(input_record, normalized, queries, candidates,
                              match, conf, trace, reason, validations,
                              validation_status)

    # --- Touchpoints 4 + 5 + routing, assembled into an MMVState ------
    def _assemble(self, input_record, normalized, queries, candidates, match,
                  confidence, trace, reason, validations, validation_status) -> dict:
        tier = confidence_tier(confidence)
        verifier_result = None
        final_decision = "review"

        if tier == "auto" and match:
            verifier_result = self._verify(normalized, match, candidates)
            final_decision = "approved" if verifier_result["passed"] else "review"
        elif tier == "review":
            final_decision = "review"
        else:
            final_decision = "rejected"

        explanation = self._explain(final_decision, normalized, match, reason,
                                    verifier_result)

        return {
            "input_record": input_record,
            "normalized_record": normalized,
            "search_queries": queries,
            "candidate_records": candidates,
            "selected_match": match or {},
            "reasoning_decision": {"match": match, "confidence": confidence,
                                   "reason": reason},
            "reasoning_trace": trace,
            "validation_status": validation_status,
            "verifier_result": verifier_result,
            "confidence": confidence,
            "explanation": explanation,
            "final_decision": final_decision,
            "step_count": len(trace),
            "llm_call_log": [],
        }

    def _verify(self, normalized: dict, match: dict, candidates: list) -> dict:
        """Adversarial check: is there a rival differing only on an omitted attr?"""
        for cand in candidates:
            if cand.get("mmv_id") == match.get("mmv_id"):
                continue
            if validate(normalized, cand)["validation_status"] != "pass":
                continue
            diff = _differing_fields(match, cand)
            omitted = [d for d in diff if normalized.get(d) in (None, "")]
            if diff and set(diff) == set(omitted):
                return {
                    "passed": False, "verdict": "fail", "ok": True,
                    "concern": (f"{cand['mmv_id']} is an equally valid match that "
                                f"differs only on {', '.join(diff)} — attributes the "
                                f"input never specified."),
                }
        return {
            "passed": True, "verdict": "pass", "ok": True,
            "concern": "No equally-valid rival candidate found; the match is unique.",
        }

    def _explain(self, decision, normalized, match, reason, verifier_result) -> str:
        label = _describe(normalized)
        if decision == "approved" and match:
            text = (f"Approved: '{label}' was matched to {_describe(match)} "
                    f"({match.get('mmv_id')}); it cleared hard-rule validation and "
                    f"passed the adversarial verifier.")
        elif decision == "rejected":
            text = f"Rejected: no catalogue record acceptably matches '{label}'. {reason}"
        elif match:
            text = (f"Flagged for review: the best candidate for '{label}' was "
                    f"{_describe(match)} ({match.get('mmv_id')}), but confidence was "
                    f"insufficient to auto-approve. {reason}")
        else:
            text = f"Flagged for review: '{label}' could not be confidently matched."
        if verifier_result and not verifier_result.get("passed"):
            text += f" Verifier concern: {verifier_result.get('concern')}"
        return text


def _canon_trans(word: Optional[str]) -> Optional[str]:
    if not word:
        return None
    w = word.lower()
    if w in ("amt", "cvt", "ivt", "dct"):
        return w.upper()
    if w == "automatic":
        return "Automatic"
    if w == "manual":
        return "Manual"
    return word.capitalize()


def _step(step: int, thought: str, action: str, action_input: dict, observation) -> dict:
    return {"step": step, "thought": thought, "action": action,
            "action_input": action_input, "observation": observation}


def _differing_fields(a: dict, b: dict) -> list:
    fields = ("make", "model", "variant", "fuel_type", "transmission", "cc",
              "seating_capacity")
    out = []
    for f in fields:
        av, bv = a.get(f), b.get(f)
        if av is None or bv is None:
            continue
        if str(av).strip().lower() != str(bv).strip().lower():
            out.append(f)
    return out


def _describe(record: dict) -> str:
    parts = [str(record[k]) for k in
             ("make", "model", "variant", "fuel_type", "transmission")
             if record.get(k)]
    return " ".join(parts).strip() or str(record.get("raw_input") or "the input vehicle")
