"""Touchpoint 5 — Explanation.

The pipeline's human-facing output. By the time this runs the decision is already
fixed: the reasoning agent (Touchpoint 3) selected or rejected a candidate, the
validation gate and confidence router graded it, and — for top-tier matches only —
the adversarial verifier (Touchpoint 4) challenged it. This touchpoint synthesizes
across ALL of that upstream state (candidates considered, selected match,
validation result, verifier outcome, final decision) into a short,
reviewer-readable summary of WHY the decision was reached.

It is a summarization step, not a decision step: it never overturns the
final_decision it is handed. Because there is no reasoning or tool use, it runs on
the cheapest/smallest model tier (config.EXPLANATION_MODEL / gemini-flash-lite).

If the LLM call fails for any reason, we fall back to a deterministic,
template-built explanation rather than crashing the pipeline: the decision is
already made and a broken explainer must not sink an otherwise-complete record.

Public surface:
- ``explain(state)`` — LangGraph node: reads MMVState, returns a partial update
  ``{"explanation": ..., "llm_call_log": ...}``.
- ``explain_decision(...)`` — helper returning just the explanation string, for
  scripts/tests that don't build an MMVState.
"""

from __future__ import annotations

import json
from typing import Optional

import config
from logging_config import get_logger
from prompts.explanation_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from services.llm_service import LLMService
from state import MMVState

# Attributes surfaced to the explainer for each candidate (compact; the model
# does not need the retrieval score internals, only the discriminating fields).
_CANDIDATE_FIELDS = (
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

# Retrieval bookkeeping fields — noise in a human explanation.
_SCORE_FIELDS = ("fuzzy_score", "embedding_score", "combined_score")


def _make_explanation_service() -> LLMService:
    """A fresh LLMService on the cheap/small tier, per call.

    Per-call (not a module singleton) so ``call_log`` stays scoped to a single
    record — mirroring how the reasoning and verifier touchpoints manage theirs.
    """
    return LLMService(
        model=config.EXPLANATION_MODEL,
        temperature=config.EXPLANATION_TEMPERATURE,
    )


def _slim_candidate(candidate: dict) -> dict:
    return {k: candidate.get(k) for k in _CANDIDATE_FIELDS if k in candidate}


def _slim_match(record: Optional[dict]) -> Optional[dict]:
    if not record:
        return None
    return {k: v for k, v in record.items() if k not in _SCORE_FIELDS}


def _describe_vehicle(record: dict) -> str:
    """A compact one-line label for a normalized/candidate record."""
    parts = [
        str(record[k])
        for k in ("make", "model", "variant", "fuel_type", "transmission")
        if record.get(k)
    ]
    label = " ".join(parts).strip()
    return label or str(record.get("raw_input") or "the input vehicle")


def _fallback_explanation(
    final_decision: str,
    normalized_record: dict,
    selected_match: Optional[dict],
    reason: str,
    verifier_result: Optional[dict],
) -> str:
    """Deterministic explanation used when the LLM call/parse fails.

    Never empty: guarantees every record still carries a human-readable
    explanation even if the (cheap) explanation model is unavailable.
    """
    input_label = _describe_vehicle(normalized_record)
    decision = (final_decision or "review").lower()

    if decision == "approved" and selected_match:
        match_label = _describe_vehicle(selected_match)
        mmv_id = selected_match.get("mmv_id")
        text = (
            f"Approved: input '{input_label}' was matched to {match_label}"
            f"{f' ({mmv_id})' if mmv_id else ''}, which cleared hard-rule "
            f"validation and passed the adversarial verifier."
        )
    elif decision == "rejected":
        text = (
            f"Rejected: no catalogue record was an acceptable match for input "
            f"'{input_label}', so it was not assigned an MMV id."
        )
    else:  # review (and any unexpected value)
        if selected_match:
            match_label = _describe_vehicle(selected_match)
            mmv_id = selected_match.get("mmv_id")
            text = (
                f"Flagged for review: the best candidate for input '{input_label}' "
                f"was {match_label}{f' ({mmv_id})' if mmv_id else ''}, but "
                f"confidence was insufficient to auto-approve."
            )
        else:
            text = (
                f"Flagged for review: input '{input_label}' could not be matched "
                f"with enough confidence to decide automatically."
            )

    concern = (verifier_result or {}).get("concern")
    if verifier_result and not verifier_result.get("passed") and concern:
        text += f" Verifier concern: {concern}"

    agent_reason = (reason or "").strip()
    if agent_reason and decision != "approved":
        text += f" Reasoning agent noted: {agent_reason}"

    return text


def explain_decision(
    final_decision: str,
    normalized_record: dict,
    candidate_records: list,
    selected_match: Optional[dict],
    validation_status: str,
    confidence: object,
    reason: str = "",
    verifier_result: Optional[dict] = None,
    service: Optional[LLMService] = None,
) -> dict:
    """Synthesize a reviewer-readable explanation of the final decision.

    Returns ``{"explanation": str, "llm_call_log": list}``. On any LLM/parse
    failure it returns a deterministic fallback explanation (still non-empty) and
    whatever call records were produced.
    """
    own_service = service is None
    service = service or _make_explanation_service()

    candidates_display = json.dumps(
        [_slim_candidate(c) for c in (candidate_records or [])],
        ensure_ascii=False,
        indent=2,
    )
    if verifier_result:
        verifier_display = json.dumps(
            {
                "verdict": verifier_result.get("verdict"),
                "passed": verifier_result.get("passed"),
                "concern": verifier_result.get("concern"),
            },
            ensure_ascii=False,
        )
    else:
        verifier_display = "n/a (verifier did not run for this record)"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        final_decision=final_decision or "review",
        confidence="unknown" if confidence is None else confidence,
        validation_status=validation_status or "unknown",
        normalized_record=json.dumps(normalized_record, ensure_ascii=False, indent=2),
        candidates=candidates_display,
        selected_match=json.dumps(_slim_match(selected_match), ensure_ascii=False, indent=2),
        reason=(reason or "").strip() or "(no justification provided)",
        verifier=verifier_display,
    )

    explanation = ""
    try:
        data = service.generate_json("explanation", SYSTEM_PROMPT, user_prompt)
        explanation = str(data.get("explanation", "")).strip()
    except (ValueError, RuntimeError):
        explanation = ""

    if not explanation:
        explanation = _fallback_explanation(
            final_decision, normalized_record, selected_match, reason, verifier_result
        )

    call_log = list(service.call_log) if own_service else []
    return {"explanation": explanation, "llm_call_log": call_log}


def explain(state: MMVState) -> dict:
    """LangGraph node: summarize the final decision for a human reviewer.

    Reads the full upstream state (normalized input, candidates considered,
    selected match, validation status, verifier outcome, final_decision) and
    returns ``{"explanation": ..., "llm_call_log": ...}``. This is the final LLM
    touchpoint in the graph; the decision is already fixed before it runs.
    """
    log = get_logger("node.explain")
    normalized = state.get("normalized_record") or state.get("input_record") or {}
    decision = state.get("reasoning_decision") or {}
    log.info("explain start: final_decision=%s", state.get("final_decision", "review"))

    outcome = explain_decision(
        final_decision=state.get("final_decision", "review"),
        normalized_record=normalized,
        candidate_records=state.get("candidate_records") or [],
        selected_match=state.get("selected_match") or None,
        validation_status=state.get("validation_status", "unknown"),
        confidence=state.get("confidence"),
        reason=decision.get("reason", ""),
        verifier_result=state.get("verifier_result") or None,
    )

    log.info("explain done (%d chars)", len(outcome.get("explanation") or ""))
    prior_log = state.get("llm_call_log") or []
    return {
        "explanation": outcome["explanation"],
        "llm_call_log": list(prior_log) + outcome.get("llm_call_log", []),
    }
